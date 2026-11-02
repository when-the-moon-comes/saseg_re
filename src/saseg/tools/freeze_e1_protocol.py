from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a validation split without touching the final test cases")
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--val-cases", type=int, default=4)
    args = parser.parse_args(); payload = json.loads(args.input.read_text())
    train_ids = sorted(case["case_id"] for case in payload["cases"] if case["split"] == "train")
    if len(train_ids) <= args.val_cases: raise SystemExit("not enough training cases")
    shuffled = train_ids.copy(); random.Random(args.seed).shuffle(shuffled); val_ids = set(shuffled[:args.val_cases])
    for case in payload["cases"]:
        if case["case_id"] in val_ids: case["split"] = "val"
    payload["parent_data_contract_sha256"] = payload["data_contract_sha256"]
    payload["protocol_seed"] = args.seed; payload["validation_case_ids"] = sorted(val_ids)
    contract = {"dataset": payload["dataset"], "labels": payload["labels"],
                "cases": sorted(payload["cases"], key=lambda x: (x["split"], x["case_id"]))}
    payload["cases"] = contract["cases"]
    payload["split_counts"] = {split: sum(c["split"] == split for c in payload["cases"]) for split in ("train", "val", "test")}
    payload["data_contract_sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "splits": payload["split_counts"],
                      "validation_case_ids": sorted(val_ids), "contract": payload["data_contract_sha256"]}, indent=2))


if __name__ == "__main__": main()
