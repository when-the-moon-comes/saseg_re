from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image


SYNAPSE_LABELS = {
    0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney",
    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "aorta", 9: "inferior vena cava", 10: "portal vein and splenic vein",
    11: "pancreas", 12: "right adrenal gland", 13: "left adrenal gland",
}
AMOS_LABELS = {
    0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney",
    4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "aorta", 9: "postcava", 10: "pancreas", 11: "right adrenal gland",
    12: "left adrenal gland", 13: "duodenum", 14: "bladder",
    15: "prostate/uterus",
}
VOC_LABELS = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
              5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
              11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
              16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
CITYSCAPES_LABELS = {7: "road", 8: "sidewalk", 11: "building", 12: "wall", 13: "fence",
                     17: "pole", 19: "traffic light", 20: "traffic sign", 21: "vegetation",
                     22: "terrain", 23: "sky", 24: "person", 25: "rider", 26: "car",
                     27: "truck", 28: "bus", 31: "train", 32: "motorcycle", 33: "bicycle"}

PROTOCOLS = {
    "synapse": {"known": list(range(1, 9)), "seen_unknown": [9, 10, 11], "unseen_unknown": [12, 13]},
    "amos22": {"known": [1, 2, 3, 4, 6, 7, 8, 10], "seen_unknown": [5, 9, 13, 14],
               "unseen_unknown": [11, 12, 15]},
    "voc2012": {"known": list(range(1, 16)), "seen_unknown": [16, 17, 18], "unseen_unknown": [19, 20]},
    "cityscapes": {"known": [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26],
                   "seen_unknown": [27, 28, 31], "unseen_unknown": [32, 33]},
}


def protocol_group(value: int, protocol: dict[str, list[int]]) -> str:
    if value == 0:
        return "background"
    for group in ("known", "seen_unknown", "unseen_unknown"):
        if value in protocol[group]:
            return group
    return "ignored"


def nifti_case(image: Path, label: Path, split: str, case_id: str) -> tuple[dict[str, Any], Counter[int]]:
    image_nii, label_nii = nib.load(str(image)), nib.load(str(label))
    if image_nii.shape != label_nii.shape:
        raise ValueError(f"shape mismatch for {case_id}: {image_nii.shape} != {label_nii.shape}")
    labels = np.asanyarray(label_nii.dataobj).astype(np.int16, copy=False)
    counts = Counter({int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True))})
    active_slices = np.flatnonzero(np.any(labels != 0, axis=(0, 1))).astype(int).tolist()
    return ({
        "case_id": case_id, "split": split, "image": str(image), "label": str(label),
        "shape": list(image_nii.shape), "spacing": [float(x) for x in image_nii.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(image_nii.affine)), "slice_indices": active_slices,
        "label_values": sorted(counts), "image_bytes": image.stat().st_size, "label_bytes": label.stat().st_size,
    }, counts)


def finalize(dataset: str, cases: list[dict[str, Any]], counts: Counter[int], label_names: dict[int, str],
             output: Path, notes: list[str]) -> None:
    protocol = PROTOCOLS[dataset]
    split_counts = Counter(case["split"] for case in cases)
    group_pixels = Counter()
    total = sum(counts.values())
    label_rows = []
    case_presence = Counter()
    for case in cases:
        for value in case["label_values"]:
            case_presence[int(value)] += 1
    for value in sorted(counts):
        group = protocol_group(value, protocol)
        group_pixels[group] += counts[value]
        label_rows.append({"value": value, "semantic_class": label_names.get(value, "void/ignored"),
                           "protocol_group": group, "pixels": counts[value],
                           "pixel_fraction": counts[value] / max(total, 1),
                           "case_or_image_count": case_presence[value],
                           "case_or_image_fraction": case_presence[value] / max(len(cases), 1)})
    contract = {"dataset": dataset, "labels": {"background": 0, **protocol},
                "cases": sorted(cases, key=lambda x: (x["split"], x["case_id"]))}
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "schema_version": 1, "dataset": dataset, "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": {"background": 0, **protocol}, "split_counts": dict(split_counts),
        "data_contract_sha256": contract_hash, "cases": contract["cases"], "notes": notes,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    (output / "label_audit").mkdir(exist_ok=True)
    (output / "manifests" / f"{dataset}.json").write_text(json.dumps(payload, indent=2) + "\n")
    three_d_spacings = [case["spacing"] for case in cases if "spacing" in case]
    spacing_range = None
    if three_d_spacings:
        spacing_array = np.asarray(three_d_spacings, dtype=float)
        spacing_range = {"min": spacing_array.min(axis=0).tolist(), "max": spacing_array.max(axis=0).tolist()}
    audit = {"dataset": dataset, "case_count": len(cases), "split_counts": dict(split_counts),
             "data_contract_sha256": contract_hash,
             "label_rows": label_rows, "protocol_pixel_counts": dict(group_pixels),
             "unknown_prevalence": {g: group_pixels[g] / max(total, 1) for g in ("seen_unknown", "unseen_unknown")},
             "duplicate_case_ids": [k for k, v in Counter(c["case_id"] for c in cases).items() if v > 1],
             "orientations": dict(Counter(c.get("orientation", "2D") for c in cases)),
             "spacing_range": spacing_range,
             "slice_summary": {"total": sum((c.get("shape", [0])[-1] if "spacing" in c else 1) for c in cases),
                               "nonempty": sum((len(c.get("slice_indices", [])) if "spacing" in c else int(any(v not in (0, 255) for v in c["label_values"]))) for c in cases)},
             "notes": notes}
    (output / "label_audit" / f"{dataset}.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"dataset": dataset, "cases": len(cases), "splits": dict(split_counts),
                      "manifest": str(output / 'manifests' / f'{dataset}.json')}, sort_keys=True))


def audit_amos(root: Path, output: Path) -> None:
    base = root / "amos22" / "amos22"
    cases, counts = [], Counter()
    jobs: list[tuple[Path, Path, str, str]] = []
    for split, image_dir, label_dir in (("train", "imagesTr", "labelsTr"), ("val", "imagesVa", "labelsVa")):
        for image in sorted((base / image_dir).glob("*.nii.gz")):
            numeric_id = int(image.name.removeprefix("amos_").removesuffix(".nii.gz"))
            if numeric_id >= 500:
                continue
            label = base / label_dir / image.name
            if not label.is_file():
                raise FileNotFoundError(label)
            jobs.append((image, label, split, image.name.removesuffix(".nii.gz")))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(nifti_case, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row, current = future.result()
            cases.append(row); counts.update(current)
    finalize("amos22", cases, counts, AMOS_LABELS, output, ["Official AMOS22 train/validation directories retained.",
             "CT-only subset is frozen as case IDs below 0500: 200 train and 100 validation; 40/20 MRI cases are excluded.",
             "Protocol groups use the audited AMOS22 semantic labels from dataset.json."] )


def audit_synapse(root: Path, output: Path) -> None:
    base = root / "btcv" / "extracted"
    metadata = json.loads((base / "dataset.json").read_text())
    test_ids = {Path(item).name.removeprefix("img").removesuffix(".nii.gz") for item in metadata["test"]}
    cases, counts = [], Counter()
    for label in sorted((base / "gt_segmentations").glob("label*.nii.gz")):
        case_num = label.name.removeprefix("label").removesuffix(".nii.gz")
        image = base / f"img{case_num}.npz"
        if not image.is_file(): raise FileNotFoundError(image)
        label_nii = nib.load(str(label)); labels = np.asanyarray(label_nii.dataobj).astype(np.int16, copy=False)
        current = Counter({int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True))})
        with np.load(image) as archive:
            data = archive["data"]
            if data.ndim != 4 or data.shape[0] != 2:
                raise ValueError(f"expected [image, label] channels in {image}, got {data.shape}")
            npz_labels = data[1].astype(np.int16, copy=False)
            aligned = np.transpose(np.where(npz_labels < 0, 0, npz_labels), (2, 1, 0))
            if aligned.shape != labels.shape or not np.array_equal(aligned, labels):
                raise ValueError(f"NPZ/NIfTI label mismatch for img{case_num}")
            image_shape = list(data[0].shape)
        split = "test" if case_num in test_ids else "train"
        cases.append({"case_id": f"img{case_num}", "split": split, "image": str(image), "label": str(label),
                      "storage_format": "npz_2ch_zyx", "image_channel": 0, "embedded_label_channel": 1,
                      "shape": image_shape, "label_shape": list(label_nii.shape), "slice_axis": 0,
                      "spacing": [float(x) for x in label_nii.header.get_zooms()[:3]],
                      "orientation": "".join(nib.aff2axcodes(label_nii.affine)),
                      "slice_indices": np.flatnonzero(np.any(npz_labels > 0, axis=(1, 2))).astype(int).tolist(),
                      "label_values": sorted(current), "image_bytes": image.stat().st_size, "label_bytes": label.stat().st_size})
        counts.update(current)
    finalize("synapse", cases, counts, SYNAPSE_LABELS, output, [
        "Canonical 18 training / 12 annotated evaluation partition is read from dataset.json; no case IDs are guessed.",
        "Each NPZ stores CT intensity in channel 0 and segmentation in channel 1 using ZYX order; channel 1 was checked exactly against the NIfTI label after transpose.",
        "Corrected labels 12/13 are right/left adrenal gland, not liver tumor/kidney cyst.",
        "The 1--8 / 9--11 / 12--13 partition is retained only as a reproducible anatomical holdout split; no pathology claim is permitted."] )


def audit_voc(root: Path, output: Path) -> None:
    base = root / "VOCdevkit" / "VOC2012"
    cases, counts = [], Counter()
    for split in ("train", "val"):
        ids = (base / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()
        for case_id in ids:
            image, label = base / "JPEGImages" / f"{case_id}.jpg", base / "SegmentationClass" / f"{case_id}.png"
            values, value_counts = np.unique(np.asarray(Image.open(label)), return_counts=True)
            current = Counter({int(v): int(c) for v, c in zip(values, value_counts)})
            cases.append({"case_id": case_id, "split": split, "image": str(image), "label": str(label),
                          "shape": list(np.asarray(Image.open(label)).shape), "label_values": sorted(current),
                          "image_bytes": image.stat().st_size, "label_bytes": label.stat().st_size})
            counts.update(current)
    finalize("voc2012", cases, counts, VOC_LABELS, output, ["Official VOC2012 segmentation train/val lists retained.",
             "Label 255 is void and ignored by training/evaluation."] )


def audit_cityscapes(root: Path, output: Path) -> None:
    base = root / "cityscapes"
    cases, counts = [], Counter()
    for split in ("train", "val"):
        for label in sorted((base / "gtFine" / split).glob("*/*_gtFine_labelIds.png")):
            stem = label.name.removesuffix("_gtFine_labelIds.png")
            city = label.parent.name; image = base / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png"
            if not image.is_file(): raise FileNotFoundError(image)
            arr = np.asarray(Image.open(label)); values, value_counts = np.unique(arr, return_counts=True)
            current = Counter({int(v): int(c) for v, c in zip(values, value_counts)})
            cases.append({"case_id": stem, "split": split, "image": str(image), "label": str(label),
                          "shape": list(arr.shape), "label_values": sorted(current),
                          "image_bytes": image.stat().st_size, "label_bytes": label.stat().st_size})
            counts.update(current)
    finalize("cityscapes", cases, counts, CITYSCAPES_LABELS, output, ["Official fine train/val directories retained.",
             "Only the 19 evaluation labelIds participate; all other IDs are void/ignored."] )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen manifests and label audits for SASeg E0")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["synapse", "amos22", "voc2012", "cityscapes"],
                        default=["synapse", "amos22", "voc2012", "cityscapes"] )
    args = parser.parse_args()
    actions = {"synapse": audit_synapse, "amos22": audit_amos, "voc2012": audit_voc, "cityscapes": audit_cityscapes}
    for dataset in args.datasets:
        actions[dataset](args.data_root, args.output)


if __name__ == "__main__":
    main()
