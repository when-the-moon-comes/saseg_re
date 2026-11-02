from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from saseg.data.nifti import NiftiSliceDataset, build_slice_records
from saseg.e1_train import mapping_from_manifest
from saseg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the known-mask binarization threshold")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--protocol", choices=["A1", "A2"], default="A1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--unknown-queries", type=int, default=2)
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    mapping = mapping_from_manifest(payload)
    records = [record for record in build_slice_records(args.manifest) if record["split"] == args.split]
    if args.max_batches:
        records = records[: args.max_batches]
    dataset = NiftiSliceDataset(records, mapping, args.protocol, "eval", args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = build_model(args.variant, mapping.num_known, backbone=args.backbone, pretrained=False,
                        image_size=args.size, unknown_queries=args.unknown_queries)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"]); model.to(device).eval()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    num_known = mapping.num_known
    intersection = {t: np.zeros(num_known + 1, dtype=np.int64) for t in thresholds}
    predicted = {t: np.zeros(num_known + 1, dtype=np.int64) for t in thresholds}
    truth = np.zeros(num_known + 1, dtype=np.int64)

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"][0].cpu().numpy()
            outputs = model(image)
            valid = target >= 0
            for class_index in range(1, num_known + 1):
                true_class = (target == class_index) & valid
                truth[class_index] += np.count_nonzero(true_class)
            for threshold in thresholds:
                if args.variant in {"saseg", "saseg_generic"}:
                    probs = outputs["known_logits"][0].sigmoid()
                    best_prob, best_index = probs.max(dim=0)
                    prediction = torch.where(best_prob > threshold, best_index + 1,
                                             torch.zeros_like(best_index)).cpu().numpy()
                else:
                    prediction = outputs[0, :num_known + 1].argmax(dim=0).cpu().numpy()
                for class_index in range(1, num_known + 1):
                    predicted_class = (prediction == class_index) & valid
                    true_class = (target == class_index) & valid
                    intersection[threshold][class_index] += np.count_nonzero(predicted_class & true_class)
                    predicted[threshold][class_index] += np.count_nonzero(predicted_class)

    rows = []
    for threshold in thresholds:
        per_class = {}
        for class_index in range(1, num_known + 1):
            if truth[class_index] > 0:
                per_class[str(class_index)] = float(
                    2 * intersection[threshold][class_index] /
                    max(int(predicted[threshold][class_index] + truth[class_index]), 1))
        rows.append({"threshold": threshold,
                     "macro_dice": float(np.mean(list(per_class.values()))) if per_class else None,
                     "per_class": per_class})
    result = {"checkpoint": str(args.checkpoint), "split": args.split, "rows": rows}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
