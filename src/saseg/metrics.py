from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve


def binary_metrics(target: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    target = np.asarray(target).astype(bool).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if target.size != score.size:
        raise ValueError("target and score size mismatch")
    if target.all() or (~target).all():
        raise ValueError("AUROC/AUPR require both positive and negative pixels")
    fpr, tpr, _ = roc_curve(target, score)
    idx = np.flatnonzero(tpr >= 0.95)
    result = {"auroc": float(roc_auc_score(target, score)), "aupr": float(average_precision_score(target, score)),
              "fpr_at_tpr95": float(fpr[idx[0]]) if idx.size else 1.0}
    if threshold is not None:
        pred = score >= threshold
        tp = np.count_nonzero(pred & target)
        fp = np.count_nonzero(pred & ~target)
        fn = np.count_nonzero(~pred & target)
        result.update({"f1": float(f1_score(target, pred, zero_division=0)),
                       "iou": float(tp / max(tp + fp + fn, 1)),
                       "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1))})
    return result


def select_f1_threshold(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    target = np.asarray(target).astype(bool).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if target.size != score.size:
        raise ValueError("target and score size mismatch")
    if target.all() or (~target).all():
        return float(np.median(score)), 0.0
    precision, recall, thresholds = precision_recall_curve(target, score)
    if not thresholds.size:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), float(f1[index])


def component_metrics_per_slice(target: np.ndarray, prediction: np.ndarray, min_iou: float = 0.1) -> dict[str, float]:
    """Average connected-component metrics over slices.

    ``component_metrics`` on a stacked (N, H, W) array would merge components
    across slices, so evaluation must iterate over the first axis.
    """
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.ndim == 2:
        return component_metrics(target, prediction, min_iou)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shape mismatch")
    rows = [component_metrics(t, p, min_iou) for t, p in zip(target, prediction) if t.any() or p.any()]
    if not rows:
        return {"component_precision": 0.0, "component_recall": 0.0, "component_f1": 0.0}
    keys = ("component_precision", "component_recall", "component_f1")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def candidate_metrics(target: np.ndarray, candidate: np.ndarray, min_iou: float = 0.1) -> dict[str, float]:
    """Region-candidate diagnostics for a single unknown mask.

    ``purity`` is the fraction of candidate pixels inside the ground truth;
    ``hit`` asks whether any candidate component reaches ``min_iou``;
    ``component_recall`` is the fraction of ground-truth components matched.
    """
    import cc3d

    target = np.asarray(target, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    target_pixels = int(target.sum())
    candidate_pixels = int(candidate.sum())
    intersection = int(np.count_nonzero(target & candidate))
    union = int(np.count_nonzero(target | candidate))
    result = {
        "gt_pixels": target_pixels,
        "candidate_pixels": candidate_pixels,
        "purity": float(intersection / candidate_pixels) if candidate_pixels else 0.0,
        "coverage": float(intersection / target_pixels) if target_pixels else 0.0,
        "union_iou": float(intersection / union) if union else 1.0,
        "hit": 0.0,
        "component_recall": 0.0,
        "component_precision": 0.0,
    }
    if not target_pixels:
        result["hit"] = 1.0 if not candidate_pixels else 0.0
        result["component_recall"] = 1.0
        result["component_precision"] = 1.0 if not candidate_pixels else 0.0
        return result
    gt_cc = cc3d.connected_components(target, connectivity=8)
    cand_cc = cc3d.connected_components(candidate, connectivity=8)
    gt_ids = [int(i) for i in np.unique(gt_cc) if i != 0]
    cand_ids = [int(i) for i in np.unique(cand_cc) if i != 0]
    matched_gt: set[int] = set()
    matched_cand: set[int] = set()
    for cid in cand_ids:
        cm = cand_cc == cid
        best_iou, best_gt = 0.0, 0
        for gid in gt_ids:
            gm = gt_cc == gid
            iou = np.count_nonzero(cm & gm) / max(np.count_nonzero(cm | gm), 1)
            if iou > best_iou:
                best_iou, best_gt = float(iou), gid
        if best_iou >= min_iou:
            matched_cand.add(cid); matched_gt.add(best_gt)
    result["hit"] = 1.0 if matched_gt else 0.0
    result["component_recall"] = len(matched_gt) / len(gt_ids)
    result["component_precision"] = len(matched_cand) / max(len(cand_ids), 1)
    return result


def proposal_upper_bound(score: np.ndarray, target: np.ndarray, valid: np.ndarray,
                         top_fraction: float = 0.003) -> dict[str, float]:
    """Connected components of the top-r fraction of foreground score pixels.

    Used as an upper bound for what a perfect region selector could recover from
    the detection signal alone.
    """
    score = np.asarray(score, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return candidate_metrics(target, np.zeros_like(valid))
    values = score[valid]
    k = max(1, int(round(values.size * top_fraction)))
    threshold = float(np.partition(values, values.size - k)[values.size - k])
    proposal = valid & (score >= threshold)
    return candidate_metrics(target, proposal)


def component_metrics(target: np.ndarray, prediction: np.ndarray, min_iou: float = 0.1) -> dict[str, float]:
    import cc3d

    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.ndim != prediction.ndim:
        raise ValueError("target and prediction dimensionality mismatch")
    connectivity = 8 if target.ndim == 2 else 26
    target_cc = cc3d.connected_components(target, connectivity=connectivity)
    pred_cc = cc3d.connected_components(prediction, connectivity=connectivity)
    target_ids = np.unique(target_cc); target_ids = target_ids[target_ids != 0]
    pred_ids = np.unique(pred_cc); pred_ids = pred_ids[pred_ids != 0]
    if not target_ids.size:
        return {"component_recall": 1.0 if not pred_ids.size else 0.0, "component_precision": 1.0 if not pred_ids.size else 0.0, "component_f1": 1.0 if not pred_ids.size else 0.0}
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pid in pred_ids:
        pm = pred_cc == pid
        best = (0.0, 0)
        for tid in target_ids:
            tm = target_cc == tid
            inter = np.count_nonzero(pm & tm)
            iou = inter / max(np.count_nonzero(pm | tm), 1)
            if iou > best[0]: best = (iou, int(tid))
        if best[0] >= min_iou:
            matched_pred.add(int(pid)); matched_gt.add(best[1])
    precision = len(matched_pred) / max(len(pred_ids), 1)
    recall = len(matched_gt) / len(target_ids)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"component_precision": precision, "component_recall": recall, "component_f1": f1}
