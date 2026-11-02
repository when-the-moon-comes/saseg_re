from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_labels(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("label list cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patient-level NIfTI slice manifest")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--known", type=parse_labels, required=True)
    parser.add_argument("--seen-unknown", type=parse_labels, required=True)
    parser.add_argument("--unseen-unknown", type=parse_labels, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-slices", action="store_true")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise SystemExit("val-fraction must be in (0, 1)")
    images = sorted(args.images_dir.glob("*.nii.gz"))
    if not images:
        raise SystemExit(f"no .nii.gz images found under {args.images_dir}")
    cases = []
    for image in images:
        label = args.labels_dir / image.name
        if not label.is_file():
            raise SystemExit(f"missing matching label for {image.name}: {label}")
        case_id = image.name.removesuffix(".nii.gz")
        if args.all_slices:
            count = int(nib.load(str(image)).shape[2])
            slices = list(range(count))
        else:
            labels = np.asarray(nib.load(str(label)).dataobj)
            slices = [int(index) for index in range(labels.shape[2]) if np.any(labels[:, :, index] != 0)]
        cases.append({"case_id": case_id, "image": str(image), "label": str(label), "slice_indices": slices})
    random.Random(args.seed).shuffle(cases)
    val_count = max(1, round(len(cases) * args.val_fraction))
    for index, case in enumerate(cases):
        case["split"] = "val" if index < val_count else "train"
    payload = {
        "dataset": args.images_dir.parent.name,
        "seed": args.seed,
        "labels": {"background": 0, "known": args.known, "seen_unknown": args.seen_unknown, "unseen_unknown": args.unseen_unknown},
        "cases": cases,
        "notes": ["Case-level split generated deterministically; validate against the official split before reporting paper results.",
                  "A2 training uses seen_unknown supervision while A2 evaluation maps unseen_unknown to the unknown output."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "train": len(cases) - val_count, "val": val_count,
                      "slices": sum(len(case["slice_indices"]) for case in cases)}, indent=2))


if __name__ == "__main__":
    main()
