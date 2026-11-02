from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = {
    "synapse": {"train": 18, "test": 12},
    "amos22": {"train": 200, "val": 100},
    "voc2012": {"train": 1464, "val": 1449},
    "cityscapes": {"train": 2975, "val": 500},
}
EXPECTED_LABELS = {"synapse": set(range(14)), "amos22": set(range(16)),
                   "voc2012": set(range(21)) | {255},
                   "cityscapes": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for dataset, expected_splits in EXPECTED.items():
        manifest_path = args.root / "manifests" / f"{dataset}.json"
        audit_path = args.root / "label_audit" / f"{dataset}.json"
        if not manifest_path.is_file() or not audit_path.is_file():
            raise SystemExit(f"missing E0 output for {dataset}")
        manifest = json.loads(manifest_path.read_text())
        audit = json.loads(audit_path.read_text())
        if manifest["split_counts"] != expected_splits:
            raise SystemExit(f"{dataset} split mismatch: {manifest['split_counts']} != {expected_splits}")
        if audit["duplicate_case_ids"]:
            raise SystemExit(f"{dataset} duplicate IDs: {audit['duplicate_case_ids']}")
        observed = {row["value"] for row in audit["label_rows"]}
        missing = EXPECTED_LABELS[dataset] - observed
        if missing:
            raise SystemExit(f"{dataset} missing label values: {sorted(missing)}")
        if len(manifest["cases"]) != sum(expected_splits.values()):
            raise SystemExit(f"{dataset} case count mismatch")
        if "slice_summary" not in audit or any("case_or_image_fraction" not in row for row in audit["label_rows"]):
            raise SystemExit(f"{dataset} occurrence/empty-slice audit missing")
        summary[dataset] = {"cases": len(manifest["cases"]), "splits": expected_splits,
                            "contract_sha256": manifest["data_contract_sha256"],
                            "unknown_prevalence": audit["unknown_prevalence"],
                            "orientations": audit["orientations"], "spacing_range": audit["spacing_range"],
                            "slice_summary": audit["slice_summary"]}
    (args.root / "E0_COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
