from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from saseg.data.labels import LabelMapping
from saseg.data.nifti import NiftiSliceDataset, build_slice_records
from saseg.losses import binary_branch_loss, class_agnostic_loss, saseg_loss
from saseg.metrics import binary_metrics, component_metrics_per_slice, select_f1_threshold
from saseg.models.factory import build_model

# Several processes share one node.  Default PyTorch CPU thread pools otherwise
# create ~100 threads per process and starve the GPU of input batches.
torch.set_num_threads(int(os.environ.get("SASEG_TORCH_THREADS", "1")))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class CaseGroupedSampler(torch.utils.data.Sampler[int]):
    """Shuffle cases every epoch while keeping slices of a case contiguous.

    Contiguous access keeps each worker's volume cache warm, which matters when
    several training processes share one node and the cache must stay small.
    """

    def __init__(self, records: list[dict[str, Any]], seed: int = 0, shuffle: bool = True):
        self.groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            self.groups.setdefault(str(record["case_id"]), []).append(index)
        self.case_ids = list(self.groups)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        case_ids = self.case_ids.copy()
        if self.shuffle:
            generator.shuffle(case_ids)
        for case_id in case_ids:
            yield from self.groups[case_id]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.groups.values())


IGNORE_LABELS = {
    "voc2012": (255,),
    "cityscapes": (1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30),
}


def mapping_from_manifest(manifest: dict[str, Any]) -> LabelMapping:
    labels = manifest["labels"]
    ignore = labels.get("ignore")
    if ignore is None:
        ignore = IGNORE_LABELS.get(str(manifest.get("dataset", "")).lower(), ())
    return LabelMapping(tuple(labels["known"]), tuple(labels["seen_unknown"]),
                        tuple(labels["unseen_unknown"]), int(labels.get("background", 0)),
                        tuple(int(value) for value in ignore))


def per_pixel_loss(logits: torch.Tensor, target: torch.Tensor, num_known: int, supervise_unknown: bool) -> torch.Tensor:
    mapped = target.clone()
    if not supervise_unknown:
        mapped[mapped == num_known + 1] = -100
    ce = F.cross_entropy(logits, mapped, ignore_index=-100)
    valid = mapped.ne(-100)
    probs = logits.softmax(dim=1)
    dice_terms = []
    max_class = num_known + (1 if supervise_unknown else 0)
    for class_index in range(1, max_class + 1):
        truth = mapped.eq(class_index) & valid
        pred = probs[:, class_index] * valid
        numerator = 2 * (pred * truth).sum(dim=(-2, -1)) + 1e-6
        denominator = pred.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1e-6
        dice_terms.append(1 - (numerator / denominator).mean())
    return ce + torch.stack(dice_terms).mean()


def prediction_scores(variant: str, outputs: Any, num_known: int) -> dict[str, torch.Tensor]:
    if variant in {"saseg", "saseg_generic"}:
        known_logits = outputs["known_logits"]
        neg_max_prob = 1 - known_logits.sigmoid().amax(dim=1)
        scores = {"msp": neg_max_prob, "energy": -torch.logsumexp(known_logits, dim=1),
                  "maxlogit": -known_logits.amax(dim=1)}
        # Per-pixel support score of the best-known query (prototype-sphere signal).
        best = known_logits.sigmoid().argmax(dim=1, keepdim=True)
        scores["support"] = torch.gather(outputs["support_score"], 1, best.flatten(1)).view_as(neg_max_prob)
        if variant == "saseg":
            scores["unknown_union"] = outputs["unknown_union"]
            scores["ensemble"] = 0.8 * neg_max_prob + 0.2 * outputs["unknown_union"]
        else:
            scores["objectness"] = outputs["unknown_union"]
        return scores
    if variant == "saseg_classagnostic":
        logits = outputs["pixel_logits"]
        known_logits = outputs["known_logits"]
        return {"unknown_class": logits.softmax(dim=1)[:, num_known + 1],
                "msp": 1 - known_logits.softmax(dim=1).amax(dim=1),
                "energy": -torch.logsumexp(known_logits, dim=1),
                "maxlogit": -known_logits.amax(dim=1)}
    if variant == "binary_branch":
        known_logits = outputs["known_logits"][:, 1:]
        return {"unknown_branch": outputs["unknown_union"],
                "msp": 1 - known_logits.softmax(dim=1).amax(dim=1),
                "energy": -torch.logsumexp(known_logits, dim=1),
                "maxlogit": -known_logits.amax(dim=1)}
    logits = outputs
    known_logits = logits[:, 1:num_known + 1]
    scores = {"msp": 1 - logits.softmax(dim=1).amax(dim=1),
              "energy": -torch.logsumexp(logits, dim=1), "maxlogit": -logits.amax(dim=1)}
    if variant == "perpixel_k1":
        scores["unknown_class"] = logits.softmax(dim=1)[:, num_known + 1]
    return scores


@torch.inference_mode()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, variant: str, num_known: int,
                        max_batches: int | None, compute_known_dice: bool = True) -> dict[str, Any]:
    model.eval(); targets: list[np.ndarray] = []; foregrounds: list[np.ndarray] = []; score_lists: dict[str, list[np.ndarray]] = {}
    case_ids: list[str] = []; slices: list[int] = []
    intersection = np.zeros(num_known + 1, dtype=np.int64) if compute_known_dice else None
    predicted_pixels = np.zeros(num_known + 1, dtype=np.int64) if compute_known_dice else None
    true_pixels = np.zeros(num_known + 1, dtype=np.int64) if compute_known_dice else None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches: break
        image = batch["image"].to(device, non_blocking=True)
        target_tensor = batch["target"].to(device, non_blocking=True)
        target = target_tensor.cpu().numpy()
        outputs = model(image); scores = prediction_scores(variant, outputs, num_known)
        if any(not torch.isfinite(score).all() for score in scores.values()):
            raise RuntimeError(f"non-finite validation score in batch {batch_index}")
        if compute_known_dice:
            prediction = known_segmentation_prediction(outputs, variant, num_known).cpu().numpy()
            valid = target >= 0
            for class_index in range(1, num_known + 1):
                predicted_class = (prediction == class_index) & valid
                true_class = (target == class_index) & valid
                intersection[class_index] += np.count_nonzero(predicted_class & true_class)
                predicted_pixels[class_index] += np.count_nonzero(predicted_class)
                true_pixels[class_index] += np.count_nonzero(true_class)
        targets.append(target == num_known + 1)
        foregrounds.append((target != 0) & (target != -100))
        for name, score in scores.items(): score_lists.setdefault(name, []).append(score.float().cpu().numpy())
        case_ids.extend(batch["case_id"]); slices.extend(int(x) for x in batch["slice"])
    target_array = np.concatenate(targets).astype(np.uint8)
    model.train()
    return {"target": target_array, "foreground": np.concatenate(foregrounds),
            "scores": {n: np.concatenate(c).astype(np.float32) for n, c in score_lists.items()},
            "case_id": np.asarray(case_ids), "slice": np.asarray(slices),
            "known_dice": known_dice_from_counts(intersection, predicted_pixels, true_pixels, num_known)
            if compute_known_dice else None}


def _histogram_ranking(pos_hist: np.ndarray, neg_hist: np.ndarray) -> dict[str, float]:
    """AUROC / AUPR / FPR@TPR95 from per-bin positive and negative counts."""
    total_pos = float(pos_hist.sum())
    total_neg = float(neg_hist.sum())
    if total_pos <= 0 or total_neg <= 0:
        return {}
    cum_pos = np.cumsum(pos_hist[::-1])
    cum_neg = np.cumsum(neg_hist[::-1])
    tpr = cum_pos / total_pos
    fpr = cum_neg / total_neg
    # cum_* run from the highest-score bin downwards, so (0,0) followed by
    # (fpr, tpr) traces the ROC curve as the threshold decreases.
    tpr_pts = np.concatenate([[0.0], tpr])
    fpr_pts = np.concatenate([[0.0], fpr])
    auroc = float(np.trapz(tpr_pts, fpr_pts))
    idx = int(np.searchsorted(tpr, 0.95))
    fpr95 = float(fpr[idx]) if idx < fpr.size else 1.0
    precision = cum_pos / np.maximum(cum_pos + cum_neg, 1.0)
    recall_prev = np.concatenate([[0.0], tpr[:-1]])
    aupr = float(np.sum((tpr - recall_prev) * precision))
    return {"auroc": auroc, "aupr": aupr, "fpr_at_tpr95": fpr95}


def evaluate_streaming(model: torch.nn.Module, val_loader: DataLoader, test_loader: DataLoader,
                       device: torch.device, variant: str, num_known: int, output_dir: Path,
                       compute_components: bool = False, bins: int = 4096,
                       component_scores: set[str] | None = None) -> dict[str, Any]:
    """Memory-bounded evaluation for large test splits (e.g. AMOS22: 11,256 slices).

    Validation scores stay in RAM as float16 for exact threshold selection.  Test
    scores are reduced on the fly into per-bin positive/negative histograms, so
    memory is independent of the number of test slices.
    """
    model.eval()
    val_scores: dict[str, list[np.ndarray]] = {}
    val_targets: list[np.ndarray] = []
    val_foreground: list[np.ndarray] = []
    intersection = np.zeros(num_known + 1, dtype=np.int64)
    predicted_pixels = np.zeros(num_known + 1, dtype=np.int64)
    true_pixels = np.zeros(num_known + 1, dtype=np.int64)
    with torch.inference_mode():
        for batch in val_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].cpu().numpy()
            outputs = model(image)
            scores = prediction_scores(variant, outputs, num_known)
            prediction = known_segmentation_prediction(outputs, variant, num_known).cpu().numpy()
            valid = target >= 0
            for class_index in range(1, num_known + 1):
                predicted_class = (prediction == class_index) & valid
                true_class = (target == class_index) & valid
                intersection[class_index] += np.count_nonzero(predicted_class & true_class)
                predicted_pixels[class_index] += np.count_nonzero(predicted_class)
                true_pixels[class_index] += np.count_nonzero(true_class)
            val_targets.append(target == num_known + 1)
            val_foreground.append((target != 0) & (target != -100))
            for name, score in scores.items():
                val_scores.setdefault(name, []).append(score.half().cpu().numpy())
    val_target = np.concatenate(val_targets).astype(bool)
    val_fg = np.concatenate(val_foreground)
    val_score_arrays = {name: np.concatenate(parts) for name, parts in val_scores.items()}
    known_val = known_dice_from_counts(intersection, predicted_pixels, true_pixels, num_known)

    thresholds: dict[str, float] = {}
    edges: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for name, scores in val_score_arrays.items():
        flat = scores.astype(np.float32, copy=False).reshape(-1)
        target_flat = val_target.reshape(-1)
        if flat.size > 5_000_000:
            index = rng.choice(flat.size, 5_000_000, replace=False)
            sample_scores, sample_target = flat[index], target_flat[index]
        else:
            sample_scores, sample_target = flat, target_flat
        threshold, _ = select_f1_threshold(sample_target, sample_scores)
        thresholds[name] = float(threshold)
        finite = sample_scores[np.isfinite(sample_scores)]
        low, high = np.quantile(finite, [0.001, 0.999]) if finite.size else (0.0, 1.0)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(finite.min()), float(finite.max()) + 1e-6
        edges[name] = np.concatenate([[-np.inf], np.linspace(low, high, bins + 1)[1:-1], [np.inf]])

    pos_hist = {name: np.zeros(bins, dtype=np.int64) for name in thresholds}
    neg_hist = {name: np.zeros(bins, dtype=np.int64) for name in thresholds}
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in thresholds}
    component_rows: dict[str, list[dict[str, float]]] = {name: [] for name in thresholds}
    test_intersection = np.zeros(num_known + 1, dtype=np.int64)
    test_predicted = np.zeros(num_known + 1, dtype=np.int64)
    test_true = np.zeros(num_known + 1, dtype=np.int64)
    with torch.inference_mode():
        for batch in test_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].cpu().numpy()
            outputs = model(image)
            scores = prediction_scores(variant, outputs, num_known)
            prediction = known_segmentation_prediction(outputs, variant, num_known).cpu().numpy()
            valid = target >= 0
            for class_index in range(1, num_known + 1):
                predicted_class = (prediction == class_index) & valid
                true_class = (target == class_index) & valid
                test_intersection[class_index] += np.count_nonzero(predicted_class & true_class)
                test_predicted[class_index] += np.count_nonzero(predicted_class)
                test_true[class_index] += np.count_nonzero(true_class)
            target_unknown = (target == num_known + 1)
            foreground = (target != 0) & (target != -100)
            for name, score in scores.items():
                score_np = score.float().cpu().numpy()
                name_edges = edges[name]
                pos_hist[name] += np.histogram(score_np[foreground & target_unknown], bins=name_edges)[0]
                neg_hist[name] += np.histogram(score_np[foreground & ~target_unknown], bins=name_edges)[0]
                pred = score_np >= thresholds[name]
                counts[name]["tp"] += int(np.count_nonzero(pred & target_unknown))
                counts[name]["fp"] += int(np.count_nonzero(pred & ~target_unknown))
                counts[name]["fn"] += int(np.count_nonzero(~pred & target_unknown))
                if compute_components and (component_scores is None or name in component_scores):
                    component_rows[name].append(component_metrics_per_slice(target_unknown, pred))
    results: dict[str, Any] = {"known_segmentation": {
        "validation": known_val,
        "test": known_dice_from_counts(test_intersection, test_predicted, test_true, num_known)}}
    for name in thresholds:
        ranking = _histogram_ranking(pos_hist[name], neg_hist[name])
        tp, fp, fn = counts[name]["tp"], counts[name]["fp"], counts[name]["fn"]
        metrics: dict[str, Any] = {"threshold": thresholds[name], **ranking}
        metrics.update({"f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                        "iou": float(tp / max(tp + fp + fn, 1)),
                        "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1)),
                        "ranking_pixels": int(pos_hist[name].sum() + neg_hist[name].sum())})
        if compute_components and component_rows[name]:
            keys = ("component_precision", "component_recall", "component_f1")
            metrics.update({key: float(np.mean([row[key] for row in component_rows[name]])) for key in keys})
        val_flat = val_score_arrays[name].astype(np.float32).reshape(-1)
        fg_flat = val_fg.reshape(-1)
        target_flat = val_target.reshape(-1)
        val_pos = np.histogram(val_flat[fg_flat & target_flat], bins=edges[name])[0]
        val_neg = np.histogram(val_flat[fg_flat & ~target_flat], bins=edges[name])[0]
        val_pred = val_flat >= thresholds[name]
        vt = int(np.count_nonzero(val_pred & target_flat))
        vf = int(np.count_nonzero(val_pred & ~target_flat))
        vn = int(np.count_nonzero(~val_pred & target_flat))
        validation = {"threshold": thresholds[name], **_histogram_ranking(val_pos, val_neg),
                      "f1": float(2 * vt / max(2 * vt + vf + vn, 1)),
                      "iou": float(vt / max(vt + vf + vn, 1)),
                      "precision": float(vt / max(vt + vf, 1)),
                      "recall": float(vt / max(vt + vn, 1))}
        results[name] = {"validation": validation, "test": metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    model.train()
    return results


def safe_metrics(target: np.ndarray, score: np.ndarray, threshold: float,
                 foreground: np.ndarray | None = None) -> dict[str, Any]:
    """Ranking metrics on foreground pixels, thresholded metrics on all pixels.

    AUROC/AUPR/FPR@TPR95 follow the paper's ``*_fg`` definition: background is
    excluded, otherwise the dominant background class reverses the ranking.
    F1/IoU stay pixel-wise over the whole image so they measure the actual mask.
    """
    target = np.asarray(target).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    region = np.ones_like(target, dtype=bool) if foreground is None else np.asarray(foreground, dtype=bool)
    result: dict[str, Any] = {"threshold": float(threshold)}
    if np.unique(target[region]).size >= 2:
        ranking = binary_metrics(target[region], score[region])
        result.update({key: ranking[key] for key in ("auroc", "aupr", "fpr_at_tpr95")})
        result["ranking_pixels"] = int(region.sum())
    else:
        result["status"] = "single_class_foreground"
    pred = score >= threshold
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    result.update({"f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                   "iou": float(tp / max(tp + fp + fn, 1)),
                   "precision": float(tp / max(tp + fp, 1)),
                   "recall": float(tp / max(tp + fn, 1)),
                   "positive_pixels": int(target.sum()), "total_pixels": int(target.size)})
    return result


def known_segmentation_prediction(outputs: Any, variant: str, num_known: int) -> torch.Tensor:
    if variant in {"saseg", "saseg_generic"}:
        # The background query is an unsupervised context token (see Sec. IV-C);
        # letting it compete in the argmax lets a random logit steal known pixels.
        # Known classes are sigmoid masks; fall back to background where none fires.
        known_probs = outputs["known_logits"].sigmoid()
        best_prob, best_index = known_probs.max(dim=1)
        return torch.where(best_prob > 0.5, best_index + 1, torch.zeros_like(best_index))
    if variant == "saseg_classagnostic":
        return outputs["pixel_logits"][:, :num_known + 1].argmax(dim=1)
    if variant == "binary_branch":
        return outputs["known_logits"].argmax(dim=1)
    return outputs[:, :num_known + 1].argmax(dim=1)


def known_dice_from_counts(intersection: np.ndarray, predicted: np.ndarray, truth: np.ndarray,
                           num_known: int) -> dict[str, Any]:
    per_class: dict[str, float] = {}
    for class_index in range(1, num_known + 1):
        if truth[class_index] > 0:
            per_class[str(class_index)] = float(
                2 * intersection[class_index] / max(int(predicted[class_index] + truth[class_index]), 1)
            )
    if not per_class:
        raise RuntimeError("validation split contains no known foreground class pixels")
    return {"macro_dice": float(np.mean(list(per_class.values()))), "per_class_dice": per_class,
            "classes_present": len(per_class)}


class ValidationAnomalyMonitor:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.zero_known_dice_epochs = 0
        self.zero_unknown_f1_epochs = 0

    def update(self, metrics: dict[str, Any], epoch: int) -> list[dict[str, Any]]:
        known_dice = metrics["known_segmentation"]["validation"]["macro_dice"]
        detection_metrics = [value for key, value in metrics.items() if key != "known_segmentation"]
        f1_values = [item["validation"].get("f1") for item in detection_metrics]
        all_unknown_f1_zero = bool(f1_values) and all(value is not None and value <= 0 for value in f1_values)
        self.zero_known_dice_epochs = self.zero_known_dice_epochs + 1 if known_dice <= 0 else 0
        self.zero_unknown_f1_epochs = self.zero_unknown_f1_epochs + 1 if all_unknown_f1_zero else 0
        events = []
        for reason, active, count in (
            ("validation_known_dice_zero", known_dice <= 0, self.zero_known_dice_epochs),
            ("validation_unknown_f1_zero", all_unknown_f1_zero, self.zero_unknown_f1_epochs),
        ):
            if active:
                events.append({"type": "stop" if count >= self.patience else "warning", "reason": reason,
                               "epoch": epoch, "consecutive_validation_epochs": count, "patience": self.patience})
        return events


def evaluate_splits(model: torch.nn.Module, val_loader: DataLoader, test_loader: DataLoader | None,
                    device: torch.device, variant: str, num_known: int, max_batches: int | None,
                    output_dir: Path, save_predictions: bool = False,
                    compute_components: bool = False,
                    component_scores: set[str] | None = None) -> dict[str, Any]:
    val = collect_predictions(model, val_loader, device, variant, num_known, max_batches)
    test = collect_predictions(model, test_loader, device, variant, num_known, max_batches) if test_loader is not None else None
    if max_batches is None and np.unique(val["target"]).size < 2:
        raise RuntimeError("full validation split lacks positive or negative unknown pixels")
    if test is not None and max_batches is None and np.unique(test["target"]).size < 2:
        raise RuntimeError("full test split lacks positive or negative unknown pixels")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"known_segmentation": {"validation": val["known_dice"],
                                                      "test": test["known_dice"] if test is not None else None}}
    for name, val_score in val["scores"].items():
        threshold, _ = select_f1_threshold(val["target"], val_score)
        results[name] = {"validation": safe_metrics(val["target"], val_score, threshold, val["foreground"])}
        test_score = None
        if test is not None:
            test_score = test["scores"][name]
            test_metrics = safe_metrics(test["target"], test_score, threshold, test["foreground"])
            if compute_components and (component_scores is None or name in component_scores):
                test_prediction = (test_score >= threshold).astype(np.uint8)
                test_metrics.update(component_metrics_per_slice(test["target"], test_prediction.astype(bool)))
            results[name]["test"] = test_metrics
        if save_predictions:
            splits = [("val", val, val_score)] + ([("test", test, test_score)] if test is not None else [])
            for split, data, score in splits:
                np.savez_compressed(output_dir / f"{split}_{name}.npz", target=data["target"], score=score,
                    foreground=data["foreground"], prediction=(score >= threshold).astype(np.uint8),
                    case_id=data["case_id"], slice=data["slice"])
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n"); return results
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n"); return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 reproducible Synapse training runner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=["perpixel_k", "perpixel_k1", "saseg", "saseg_generic",
                                              "saseg_classagnostic", "binary_branch"], required=True)
    parser.add_argument("--unknown-queries", type=int, default=2)
    parser.add_argument("--spheres-per-class", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--protocol", choices=["A1", "A2"], default="A1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int); parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--size", type=int, default=224); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0); parser.add_argument("--volume-cache", type=int, default=1)
    parser.add_argument("--volume-cache-mb", type=int, default=256); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path); parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate a checkpoint with the requested protocol.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip the full final evaluation (smoke runs).")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint used with --eval-only.")
    parser.add_argument("--skip-components", action="store_true", help="Skip connected-component metrics.")
    parser.add_argument("--skip-predictions", action="store_true", help="Do not write prediction npz files.")
    parser.add_argument("--stream-eval", action="store_true",
                        help="Memory-bounded histogram evaluation for very large test splits.")
    parser.add_argument("--component-score", default=None,
                        help="Comma-separated score names for component metrics (default: all).")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-start-epoch", type=int, default=10)
    parser.add_argument("--anomaly-patience", type=int, default=5)
    parser.add_argument("--max-consecutive-skipped-updates", type=int, default=20)
    parser.add_argument("--max-training-loss", type=float, default=100.0)
    parser.add_argument("--max-consecutive-high-loss", type=int, default=10)
    parser.add_argument("--accumulation-steps", type=int, default=1); parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224.ms_in22k")
    parser.add_argument("--pretrained", action="store_true"); parser.add_argument("--augment", action="store_true")
    args = parser.parse_args(); seed_everything(args.seed)
    component_scores = ({item.strip() for item in args.component_score.split(",") if item.strip()}
                        if args.component_score else None)
    if min(args.early_stop_patience, args.anomaly_patience, args.max_consecutive_skipped_updates,
           args.max_consecutive_high_loss) < 1 or args.max_training_loss <= 0:
        raise ValueError("patience limits must be positive")
    if args.early_stop_min_delta < 0 or args.early_stop_start_epoch < 0:
        raise ValueError("early stopping delta/start epoch cannot be negative")
    manifest = json.loads(args.manifest.read_text()); mapping = mapping_from_manifest(manifest)
    records = build_slice_records(args.manifest)
    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]; test_records = [r for r in records if r["split"] == "test"]
    if not train_records or not val_records or not test_records: raise ValueError("manifest must contain train, val, and test slices")
    train_set = NiftiSliceDataset(train_records, mapping, args.protocol, "train", args.size, augment=args.augment,
                                  volume_cache=args.volume_cache, volume_cache_mb=args.volume_cache_mb)
    val_set = NiftiSliceDataset(val_records, mapping, args.protocol, "eval", args.size,
                                volume_cache=args.volume_cache, volume_cache_mb=args.volume_cache_mb)
    test_set = NiftiSliceDataset(test_records, mapping, args.protocol, "eval", args.size,
                                 volume_cache=args.volume_cache, volume_cache_mb=args.volume_cache_mb)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    # Validation/test run once per epoch; spawning workers for them multiplies
    # host memory across a shared node without a meaningful speedup.
    eval_kwargs = {"num_workers": 0, "pin_memory": True}
    train_sampler = CaseGroupedSampler(train_records, seed=args.seed, shuffle=True)
    train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, **eval_kwargs)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, **eval_kwargs)
    device = torch.device(args.device); num_known = mapping.num_known
    if args.unknown_queries < 0:
        raise ValueError("unknown-queries cannot be negative")
    model: torch.nn.Module = build_model(args.variant, num_known, backbone=args.backbone,
                                         pretrained=args.pretrained, image_size=args.size,
                                         unknown_queries=args.unknown_queries,
                                         spheres_per_class=args.spheres_per_class,
                                         decoder_layers=args.decoder_layers)
    model.to(device)
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"]); model.eval()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_config.json").write_text(json.dumps(
            {**vars(args), "manifest": str(args.manifest), "output_dir": str(args.output_dir),
             "checkpoint": str(args.checkpoint), "manifest_contract": manifest.get("data_contract_sha256"),
             "train_slices": len(train_set), "val_slices": len(val_set), "test_slices": len(test_set),
             "parameters": sum(p.numel() for p in model.parameters())}, default=str, indent=2) + "\n")
        if args.stream_eval:
            metrics = evaluate_streaming(model, val_loader, test_loader, device, args.variant, num_known,
                                         args.output_dir / "final",
                                         compute_components=not args.skip_components,
                                         component_scores=component_scores)
        else:
            metrics = evaluate_splits(model, val_loader, test_loader, device, args.variant, num_known,
                                      args.max_val_batches, args.output_dir / "final",
                                      save_predictions=not args.skip_predictions,
                                      compute_components=not args.skip_components,
                                      component_scores=component_scores)
        (args.output_dir / "COMPLETE").write_text(json.dumps(
            {"mode": "eval_only", "checkpoint": str(args.checkpoint), "protocol": args.protocol}, indent=2) + "\n")
        print(json.dumps({"type": "eval_only", "checkpoint": str(args.checkpoint), "metrics": metrics}), flush=True)
        return
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": args.backbone_lr},
                                   {"params": other_params, "lr": args.lr}], weight_decay=1e-4)
    total_updates = max(1, (len(train_loader) * args.epochs + args.accumulation_steps - 1) // args.accumulation_steps)
    def lr_factor(update: int) -> float:
        if update < args.warmup_steps:
            return max(update + 1, 1) / max(args.warmup_steps, 1)
        progress = min(1.0, (update - args.warmup_steps) / max(total_updates - args.warmup_steps, 1))
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True); start_epoch = 0; step = 0
    (args.output_dir / "run_config.json").write_text(json.dumps({**vars(args), "manifest": str(args.manifest),
        "output_dir": str(args.output_dir), "resume": str(args.resume) if args.resume else None,
        "manifest_contract": manifest.get("data_contract_sha256"), "train_slices": len(train_set),
        "val_slices": len(val_set), "test_slices": len(test_set),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, default=str, indent=2) + "\n")
    best_val_dice = -math.inf; epochs_without_improvement = 0; skipped_update_streak = 0
    high_loss_streak = 0
    anomaly_monitor = ValidationAnomalyMonitor(args.anomaly_patience)
    anomaly_path = args.output_dir / "anomalies.jsonl"
    stop_reason: str | None = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu"); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]);
        if "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state: scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1; step = int(state["step"])
        best_val_dice = float(state.get("best_val_known_dice", -math.inf))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        monitor_state = state.get("validation_anomaly_streaks", {})
        anomaly_monitor.zero_known_dice_epochs = int(monitor_state.get("zero_known_dice_epochs", 0))
        anomaly_monitor.zero_unknown_f1_epochs = int(monitor_state.get("zero_unknown_f1_epochs", 0))
    model.train(); started = time.time(); optimizer.zero_grad(set_to_none=True); update = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            if args.max_steps is not None and step >= args.max_steps: break
            image = batch["image"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                outputs = model(image)
                if args.variant == "saseg": losses = saseg_loss(outputs, target, num_known)
                elif args.variant == "saseg_generic":
                    losses = saseg_loss(outputs, target, num_known, generic_queries=True, lambda_boundary=0.0)
                elif args.variant == "saseg_classagnostic": losses = class_agnostic_loss(outputs, target, num_known)
                elif args.variant == "binary_branch": losses = binary_branch_loss(outputs, target, num_known)
                else:
                    loss = per_pixel_loss(outputs, target, num_known, args.variant == "perpixel_k1"); losses = {"total": loss}
                loss = losses["total"]
                if not torch.isfinite(loss):
                    event = {"type": "stop", "reason": "non_finite_training_loss", "epoch": epoch,
                             "step": step, "loss": float(loss.detach().float().cpu())}
                    with anomaly_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                    (args.output_dir / "STOPPED_ANOMALY.json").write_text(json.dumps(event, indent=2) + "\n")
                    raise RuntimeError(f"training stopped: {event}")
                if float(loss.detach()) > args.max_training_loss:
                    high_loss_streak += 1
                    event = {"type": "warning", "reason": "training_loss_above_limit", "epoch": epoch,
                             "step": step, "loss": float(loss.detach()),
                             "consecutive_batches": high_loss_streak, "limit": args.max_training_loss}
                    with anomaly_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                    if high_loss_streak >= args.max_consecutive_high_loss:
                        event["type"] = "stop"; event["reason"] = "training_loss_diverged"
                        with anomaly_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(event) + "\n")
                        (args.output_dir / "STOPPED_ANOMALY.json").write_text(json.dumps(event, indent=2) + "\n")
                        raise RuntimeError(f"training stopped: {event}")
                else:
                    high_loss_streak = 0
                scaled_loss = loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_update = (step + 1) % args.accumulation_steps == 0 or (args.max_steps is not None and step + 1 >= args.max_steps)
            if should_update:
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                optimizer_updated = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
                if optimizer_updated:
                    scheduler.step(); update += 1; skipped_update_streak = 0
                else:
                    skipped_update_streak += 1
                    event = {"type": "warning", "reason": "amp_optimizer_update_skipped", "epoch": epoch,
                             "step": step, "consecutive_skipped_updates": skipped_update_streak}
                    with anomaly_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                    if skipped_update_streak >= args.max_consecutive_skipped_updates:
                        event["type"] = "stop"; event["reason"] = "too_many_consecutive_skipped_updates"
                        with anomaly_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(event) + "\n")
                        (args.output_dir / "STOPPED_ANOMALY.json").write_text(json.dumps(event, indent=2) + "\n")
                        raise RuntimeError(f"training stopped: {event}")
            if args.variant in {"saseg", "saseg_generic"}: model.update_prototypes(outputs, target)  # type: ignore[attr-defined]
            print(json.dumps({"type": "train", "variant": args.variant, "epoch": epoch, "step": step,
                              "update": update, "lr": [g["lr"] for g in optimizer.param_groups],
                              "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                              "elapsed_s": time.time() - started, **{k: float(v.detach().cpu()) for k, v in losses.items()}}), flush=True)
            step += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "step": step, "seed": args.seed, "variant": args.variant, "protocol": args.protocol}, args.output_dir / "last.pt")
        if (epoch + 1) % args.val_every == 0:
            try:
                metrics = evaluate_splits(model, val_loader, None, device, args.variant, num_known,
                                          args.max_val_batches, args.output_dir / f"epoch_{epoch:03d}")
            except Exception as error:
                event = {"type": "stop", "reason": "validation_evaluation_failed", "epoch": epoch,
                         "error": repr(error)}
                with anomaly_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                (args.output_dir / "STOPPED_ANOMALY.json").write_text(json.dumps(event, indent=2) + "\n")
                raise
            print(json.dumps({"type": "validation", "epoch": epoch, "step": step, "metrics": metrics}), flush=True)
            for event in anomaly_monitor.update(metrics, epoch):
                with anomaly_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
                if event["type"] == "stop": stop_reason = event["reason"]
            val_dice = metrics["known_segmentation"]["validation"]["macro_dice"]
            if val_dice > best_val_dice + args.early_stop_min_delta:
                best_val_dice = val_dice; epochs_without_improvement = 0
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "epoch": epoch, "step": step, "seed": args.seed, "variant": args.variant,
                            "protocol": args.protocol, "best_val_known_dice": best_val_dice},
                           args.output_dir / "best.pt")
            else:
                epochs_without_improvement += 1
            if stop_reason is None and (epoch + 1) >= args.early_stop_start_epoch and epochs_without_improvement >= args.early_stop_patience:
                event = {"type": "stop", "reason": "early_stopping_no_validation_improvement",
                         "epoch": epoch, "best_val_known_dice": best_val_dice,
                         "epochs_without_improvement": epochs_without_improvement,
                         "patience": args.early_stop_patience, "min_delta": args.early_stop_min_delta}
                with anomaly_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True); stop_reason = event["reason"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "step": step, "seed": args.seed, "variant": args.variant,
                        "protocol": args.protocol, "best_val_known_dice": best_val_dice,
                        "epochs_without_improvement": epochs_without_improvement,
                        "validation_anomaly_streaks": {
                            "zero_known_dice_epochs": anomaly_monitor.zero_known_dice_epochs,
                            "zero_unknown_f1_epochs": anomaly_monitor.zero_unknown_f1_epochs,
                        }}, args.output_dir / "last.pt")
        if stop_reason is not None:
            stop_record = {"epoch": epoch, "reason": stop_reason, "best_val_known_dice": best_val_dice,
                           "epochs_without_improvement": epochs_without_improvement,
                           "anomaly_file": str(anomaly_path)}
            marker = "STOPPED_EARLY.json" if stop_reason == "early_stopping_no_validation_improvement" else "STOPPED_ANOMALY.json"
            (args.output_dir / marker).write_text(json.dumps(stop_record, indent=2) + "\n")
            break
        if args.max_steps is not None and step >= args.max_steps: break

    # Paper numbers are taken from the best-validation checkpoint, never from the
    # last epoch, so the checkpoint cannot be selected using test labels.
    best_path = args.output_dir / "best.pt"
    if best_path.is_file() and not args.skip_final_eval:
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"]); model.to(device); model.eval()
        if args.stream_eval:
            final_metrics = evaluate_streaming(model, val_loader, test_loader, device, args.variant, num_known,
                                               args.output_dir / "final",
                                               compute_components=not args.skip_components,
                                               component_scores=component_scores)
        else:
            final_metrics = evaluate_splits(model, val_loader, test_loader, device, args.variant, num_known,
                                            None, args.output_dir / "final",
                                            save_predictions=True, compute_components=True,
                                            component_scores=component_scores)
        print(json.dumps({"type": "final_evaluation", "checkpoint": str(best_path),
                          "epoch": state.get("epoch"), "metrics": final_metrics}), flush=True)


if __name__ == "__main__":
    main()
