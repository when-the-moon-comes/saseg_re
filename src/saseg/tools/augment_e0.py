from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    for manifest_path in sorted((args.root / "manifests").glob("*.json")):
        dataset = manifest_path.stem
        audit_path = args.root / "label_audit" / f"{dataset}.json"
        manifest, audit = json.loads(manifest_path.read_text()), json.loads(audit_path.read_text())
        cases = manifest["cases"]; presence = Counter()
        for case in cases:
            for value in case["label_values"]:
                presence[int(value)] += 1
        for row in audit["label_rows"]:
            value = int(row["value"]); row["case_or_image_count"] = presence[value]
            row["case_or_image_fraction"] = presence[value] / max(len(cases), 1)
        total = 0; nonempty = 0
        for case in cases:
            if "spacing" in case:
                if case.get("storage_format") == "npz_2ch_zyx":
                    slices = int(case["shape"][0])
                else:
                    slices = int(case["shape"][2])
                total += slices; nonempty += len(case.get("slice_indices", []))
            else:
                total += 1; nonempty += int(any(v not in (0, 255) for v in case["label_values"]))
        audit["slice_summary"] = {"total": total, "nonempty": nonempty, "empty": total - nonempty,
                                  "empty_fraction": (total - nonempty) / max(total, 1)}
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        print(dataset, audit["slice_summary"])


if __name__ == "__main__":
    main()
