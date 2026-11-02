from __future__ import annotations

import argparse
import json
from pathlib import Path

from saseg.data.labels import LabelMapping
from saseg.data.nifti import build_slice_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping = LabelMapping(tuple(manifest["labels"]["known"]), tuple(manifest["labels"]["seen_unknown"]), tuple(manifest["labels"]["unseen_unknown"]), int(manifest["labels"].get("background", 0)))
    records = build_slice_records(args.manifest)
    split_cases = {}
    for record in records:
        split_cases.setdefault(record["split"], set()).add(record["case_id"])
    overlap = set.intersection(*(set(value) for value in split_cases.values())) if len(split_cases) > 1 else set()
    if overlap:
        raise SystemExit(f"patient leakage across splits: {sorted(overlap)}")
    print(json.dumps({"manifest": str(args.manifest), "num_known": mapping.num_known, "records": len(records), "cases_by_split": {k: len(v) for k, v in split_cases.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
