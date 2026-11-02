from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saseg.data.nifti import NiftiSliceDataset, build_slice_records
from saseg.e1_train import known_segmentation_prediction, mapping_from_manifest, prediction_scores
from saseg.models.factory import build_model


def load_model(variant: str, num_known: int, checkpoint: Path, unknown_queries: int,
               backbone: str, size: int, device: torch.device):
    model = build_model(variant, num_known, backbone=backbone, pretrained=False,
                        image_size=size, unknown_queries=unknown_queries)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def threshold_from_metrics(path: Path, score: str) -> float:
    metrics = json.loads(path.read_text())
    return float(metrics[score]["test"]["threshold"])


def choose_indices(dataset: NiftiSliceDataset, num_known: int, count: int, min_pixels: int) -> list[int]:
    """Pick slices with enough unknown pixels, preferring distinct cases."""
    by_case: dict[str, list[tuple[int, int]]] = {}
    for index in range(len(dataset)):
        target = dataset[index]["target"].numpy()
        unknown = int(np.count_nonzero(target == num_known + 1))
        if unknown < min_pixels:
            continue
        case_id = str(dataset.records[index]["case_id"])
        by_case.setdefault(case_id, []).append((index, unknown))
    chosen: list[int] = []
    cases = sorted(by_case)
    # one slice per case first
    for case_id in cases:
        if len(chosen) >= count:
            break
        best = max(by_case[case_id], key=lambda item: item[1])
        chosen.append(best[0])
    # fill with additional slices from the richest cases
    if len(chosen) < count:
        ranked = sorted(((idx, unknown) for items in by_case.values() for idx, unknown in items),
                        key=lambda item: item[1], reverse=True)
        for idx, _ in ranked:
            if idx in chosen:
                continue
            chosen.append(idx)
            if len(chosen) >= count:
                break
    return chosen[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative figure generator")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", choices=["A1", "A2"], required=True)
    parser.add_argument("--mode", choices=["a1", "a2", "m0"], required=True)
    parser.add_argument("--saseg-checkpoint", type=Path, required=True)
    parser.add_argument("--saseg-metrics", type=Path, required=True)
    parser.add_argument("--m0-checkpoint", type=Path)
    parser.add_argument("--m0-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--min-unknown-pixels", type=int, default=200)
    parser.add_argument("--unknown-queries", type=int, default=2)
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    mapping = mapping_from_manifest(payload)
    num_known = mapping.num_known
    records = [record for record in build_slice_records(args.manifest) if record["split"] == "test"]
    dataset = NiftiSliceDataset(records, mapping, args.protocol, "eval", args.size)
    device = torch.device(args.device)
    saseg = load_model("saseg", num_known, args.saseg_checkpoint, args.unknown_queries,
                       args.backbone, args.size, device)
    m0 = None
    if args.mode == "m0":
        if not args.m0_checkpoint or not args.m0_metrics:
            raise SystemExit("m0 mode requires --m0-checkpoint and --m0-metrics")
        m0 = load_model("saseg", num_known, args.m0_checkpoint, 0, args.backbone, args.size, device)
    score_name = {"a1": "unknown_union", "a2": "ensemble", "m0": "unknown_union"}[args.mode]
    threshold = threshold_from_metrics(args.saseg_metrics, score_name)

    indices = choose_indices(dataset, num_known, args.slices, args.min_unknown_pixels)
    if not indices:
        raise SystemExit("no test slice with enough unknown pixels found")

    entries = []
    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            image = sample["image"][None].to(device)
            target = sample["target"].numpy()
            saseg_out = saseg(image)
            saseg_scores = prediction_scores("saseg", saseg_out, num_known)
            entry = {"case_id": sample["case_id"], "slice": int(sample["slice"]),
                     "image": sample["image"][0].numpy(), "target": target,
                     "score": saseg_scores[score_name][0].cpu().numpy()}
            if m0 is not None:
                m0_out = m0(image)
                m0_scores = prediction_scores("saseg", m0_out, num_known)
                entry["m0_score"] = m0_scores["msp"][0].cpu().numpy()
            entries.append(entry)

    columns = ["Image", "GT (red=unknown)", "SASeg unknown"] if args.mode in {"a1", "m0"} else (
        ["Image", "GT unknown", "Ensemble score", "Thresholded mask"])
    if args.mode == "m0":
        columns = ["Image", "GT (red=unknown)", "SASeg unknown", "M=0 MSP score"]
    fig, axes = plt.subplots(len(entries), len(columns), figsize=(3.1 * len(columns), 3.2 * len(entries)))
    if len(entries) == 1:
        axes = axes[None, :]
    for row, entry in enumerate(entries):
        image = entry["image"]
        target = entry["target"]
        gt_unknown = target == num_known + 1
        known_mask = (target >= 1) & (target <= num_known)
        saseg_mask = entry["score"] >= threshold
        axes[row, 0].imshow(image, cmap="gray"); axes[row, 0].set_title("Image", fontsize=9)
        if args.mode == "a2":
            axes[row, 1].imshow(image, cmap="gray")
            axes[row, 1].imshow(np.ma.masked_where(~gt_unknown, gt_unknown), cmap="autumn", alpha=0.85)
            axes[row, 1].set_title("GT unknown", fontsize=9)
            axes[row, 2].imshow(entry["score"], cmap="magma", vmin=0.0, vmax=1.0)
            axes[row, 2].set_title("Ensemble score", fontsize=9)
            axes[row, 3].imshow(image, cmap="gray")
            axes[row, 3].imshow(np.ma.masked_where(~saseg_mask, saseg_mask), cmap="autumn", alpha=0.45)
            axes[row, 3].set_title("Mask (t=%.2f)" % threshold, fontsize=9)
        else:
            axes[row, 1].imshow(image, cmap="gray")
            axes[row, 1].imshow(np.ma.masked_where(~known_mask, known_mask), cmap="Blues", alpha=0.35)
            axes[row, 1].imshow(np.ma.masked_where(~gt_unknown, gt_unknown), cmap="autumn", alpha=0.9)
            axes[row, 1].set_title("GT (red=unknown)", fontsize=9)
            axes[row, 2].imshow(image, cmap="gray")
            axes[row, 2].imshow(np.ma.masked_where(~saseg_mask, saseg_mask), cmap="autumn", alpha=0.45)
            axes[row, 2].set_title("SASeg unknown (t=%.2f)" % threshold, fontsize=9)
            if args.mode == "m0":
                axes[row, 3].imshow(entry["m0_score"], cmap="magma")
                axes[row, 3].set_title("M=0 MSP score", fontsize=9)
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].set_ylabel(f"{entry['case_id']} z={entry['slice']}", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(json.dumps({"output": str(args.output), "slices": [
        {"case_id": item["case_id"], "slice": item["slice"]} for item in entries]}, indent=2))


if __name__ == "__main__":
    main()
