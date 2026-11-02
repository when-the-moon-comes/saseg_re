from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from saseg.metrics import binary_metrics, component_metrics, select_f1_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen predictions without test-time threshold fitting")
    parser.add_argument("--predictions", type=Path, required=True, help="npz with target and score arrays")
    parser.add_argument("--protocol", choices=["A1", "A2"], required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--validation-predictions", type=Path, help="validation npz used only to select threshold")
    args = parser.parse_args()
    data = np.load(args.predictions)
    target, score = data["target"], data["score"]
    threshold = args.threshold
    if threshold is None and args.validation_predictions is not None:
        val = np.load(args.validation_predictions)
        threshold, _ = select_f1_threshold(val["target"], val["score"])
    result = {"protocol": args.protocol, "predictions": str(args.predictions), "threshold": threshold}
    result.update(binary_metrics(target, score, threshold))
    if "prediction" in data:
        result.update(component_metrics(data["target"], data["prediction"]))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
