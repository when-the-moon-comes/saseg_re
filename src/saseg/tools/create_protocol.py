from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def contract_hash(payload: dict) -> str:
    contract = {"dataset": payload["dataset"], "labels": payload["labels"],
                "cases": sorted(payload["cases"], key=lambda item: (item["split"], item["case_id"]))}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a leakage-free train/val/test protocol from an E0 manifest.")
    parser.add_argument("--input", type=Path, required=True, help="E0 manifest (official train/val splits)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-cases", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--official-val-as-test", action="store_true",
                        help="Use the official validation split as the final test split.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    cases = payload["cases"]

    if args.official_val_as_test:
        for case in cases:
            if case["split"] == "val":
                case["split"] = "test"

    train_ids = sorted(case["case_id"] for case in cases if case["split"] == "train")
    if len(train_ids) < 2:
        raise SystemExit("not enough training cases to hold out a validation split")
    val_count = args.val_cases if args.val_cases is not None else max(1, round(len(train_ids) * args.val_fraction))
    if val_count >= len(train_ids):
        raise SystemExit("validation holdout must leave at least one training case")
    shuffled = train_ids.copy()
    random.Random(args.seed).shuffle(shuffled)
    val_ids = set(shuffled[:val_count])
    for case in cases:
        if case["case_id"] in val_ids and case["split"] == "train":
            case["split"] = "val"

    payload["parent_data_contract_sha256"] = payload.get("data_contract_sha256")
    payload["protocol_seed"] = args.seed
    payload["validation_case_ids"] = sorted(val_ids)
    payload["cases"] = sorted(cases, key=lambda item: (item["split"], item["case_id"]))
    payload["split_counts"] = {split: sum(case["split"] == split for case in payload["cases"])
                               for split in ("train", "val", "test")}
    payload.setdefault("notes", [])
    payload["notes"] = list(payload["notes"]) + [
        "Validation cases are held out from the official training split; thresholds and early stopping use only this split.",
        "The official validation split is used as the final test split." if args.official_val_as_test
        else "Official validation cases retained; verify that no threshold is selected on the test split.",
    ]
    payload["data_contract_sha256"] = contract_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "splits": payload["split_counts"],
                      "validation_case_ids": sorted(val_ids),
                      "contract": payload["data_contract_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
