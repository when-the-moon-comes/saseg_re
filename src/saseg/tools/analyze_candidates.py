from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from saseg.data.nifti import NiftiSliceDataset, build_slice_records
from saseg.e1_train import mapping_from_manifest
from saseg.metrics import candidate_metrics, proposal_upper_bound
from saseg.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Region-candidate diagnostics on a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--protocol", choices=["A1", "A2"], default="A2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--unknown-queries", type=int, default=2)
    parser.add_argument("--spheres-per-class", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-threshold", type=float, default=0.5)
    parser.add_argument("--top-fractions", default="0.001,0.003,0.005,0.01")
    parser.add_argument("--max-slices", type=int)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    labels = payload["labels"]
    mapping = mapping_from_manifest(payload)
    records = [record for record in build_slice_records(args.manifest) if record["split"] == args.split]
    if args.max_slices:
        records = records[: args.max_slices]
    if not records:
        raise SystemExit(f"no slices in split {args.split}")
    dataset = NiftiSliceDataset(records, mapping, args.protocol, "eval", args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)

    model = build_model(args.variant, mapping.num_known, backbone=args.backbone, pretrained=False,
                        image_size=args.size, unknown_queries=args.unknown_queries,
                        spheres_per_class=args.spheres_per_class, decoder_layers=args.decoder_layers)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device).eval()

    top_fractions = [float(item) for item in args.top_fractions.split(",") if item.strip()]
    rows: list[dict] = []
    query_rows: list[dict] = []
    proposal_rows: list[dict] = []
    num_known = mapping.num_known
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"][0].cpu().numpy()
            outputs = model(image)
            gt = target == num_known + 1
            valid = target != -100
            known_logits = outputs["known_logits"]
            neg_max_prob = 1 - known_logits.sigmoid().amax(dim=1)
            scores = {}
            if "unknown_probs" in outputs and outputs["unknown_probs"].shape[1]:
                unknown_probs = outputs["unknown_probs"][0].float().cpu().numpy()
                candidates = unknown_probs >= args.candidate_threshold
                union = candidates.max(axis=0)
                rows.append({"slice": int(batch["slice"][0]),
                             **candidate_metrics(gt, union)})
                for index, candidate in enumerate(candidates):
                    query_rows.append({"slice": int(batch["slice"][0]), "query": index,
                                       **candidate_metrics(gt, candidate)})
                scores["unknown_union"] = unknown_probs.max(axis=0)
                scores["ensemble"] = (0.8 * neg_max_prob[0].float().cpu().numpy() +
                                      0.2 * scores["unknown_union"])
            else:
                scores["msp"] = neg_max_prob[0].float().cpu().numpy()
            for name, score in scores.items():
                for fraction in top_fractions:
                    proposal_rows.append({"slice": int(batch["slice"][0]), "score": name,
                                          "top_fraction": fraction,
                                          **proposal_upper_bound(score, gt, valid, fraction)})

    def aggregate(items: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
        if not items:
            return {}
        return {key: float(np.mean([item[key] for item in items])) for key in keys}
    keys = ("purity", "coverage", "union_iou", "hit", "component_recall", "component_precision")
    summary = {
        "checkpoint": str(args.checkpoint), "variant": args.variant, "protocol": args.protocol,
        "split": args.split, "slices": len(records), "candidate_threshold": args.candidate_threshold,
        "gt_pixels": int(sum(row["gt_pixels"] for row in rows)),
        "candidate_pixels": int(sum(row["candidate_pixels"] for row in rows)),
        "candidate_union": aggregate(rows, keys),
        "per_query": {str(index): aggregate([row for row in query_rows if row["query"] == index], keys)
                      for index in sorted({row["query"] for row in query_rows})},
        "proposal_upper_bound": {},
    }
    for name in sorted({row["score"] for row in proposal_rows}):
        for fraction in top_fractions:
            subset = [row for row in proposal_rows if row["score"] == name and row["top_fraction"] == fraction]
            summary["proposal_upper_bound"].setdefault(name, {})[str(fraction)] = aggregate(subset, keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "union_rows": rows, "per_query_rows": query_rows,
                                       "proposal_rows": proposal_rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
