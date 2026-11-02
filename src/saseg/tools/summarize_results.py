from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

PRIMARY_SCORE = {
    "saseg": ("unknown_union", "ensemble"),
    "saseg_m2": ("unknown_union", "ensemble"),
    "saseg_m1": ("unknown_union", "ensemble"),
    "saseg_m4": ("unknown_union", "ensemble"),
    "saseg_m0": ("msp", "msp"),
    "saseg_generic": ("msp", "msp"),
    "saseg_classagnostic": ("unknown_class", "unknown_class"),
    "binary_branch": ("unknown_branch", "unknown_branch"),
    "perpixel_k": ("msp", "msp"),
    "perpixel_k1": ("unknown_class", "unknown_class"),
}
METRIC_KEYS = ("auroc", "aupr", "fpr_at_tpr95", "f1", "iou", "precision", "recall",
               "component_precision", "component_recall", "component_f1")


def load_runs(root: Path) -> list[dict]:
    runs = []
    for metrics_path in sorted(root.rglob("final/metrics.json")):
        run_dir = metrics_path.parent.parent
        task_path = run_dir / "task.json"
        task = json.loads(task_path.read_text()) if task_path.is_file() else {}
        metrics = json.loads(metrics_path.read_text())
        runs.append({"run_dir": str(run_dir), "task": task, "metrics": metrics})
    return runs


def aggregate(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    meta: dict[tuple, dict] = {}
    for run in runs:
        task = run["task"]
        parts = Path(run["run_dir"]).parts
        dataset = task.get("dataset") or (parts[-4] if len(parts) >= 4 else "unknown")
        protocol = task.get("protocol") or (parts[-3] if len(parts) >= 3 else "unknown")
        label = task.get("label") or (parts[-2] if len(parts) >= 2 else "unknown")
        seed = task.get("seed")
        metrics = run["metrics"]
        known = metrics.get("known_segmentation", {})
        for split in ("validation", "test"):
            value = (known.get(split) or {}).get("macro_dice")
            if value is not None:
                key = (dataset, protocol, label, "known_macro_dice", split)
                grouped.setdefault(key, []).append(float(value))
                meta.setdefault(key, {"variant": task.get("variant"), "unknown_queries": task.get("unknown_queries")})
        for score, payload in metrics.items():
            if score == "known_segmentation":
                continue
            for split in ("validation", "test"):
                split_metrics = payload.get(split, {})
                for metric in METRIC_KEYS:
                    if metric in split_metrics and isinstance(split_metrics[metric], (int, float)):
                        key = (dataset, protocol, label, f"{score}.{metric}", split)
                        grouped.setdefault(key, []).append(float(split_metrics[metric]))
                        meta.setdefault(key, {"variant": task.get("variant"),
                                              "unknown_queries": task.get("unknown_queries")})
    rows = []
    for key, values in grouped.items():
        dataset, protocol, label, metric, split = key
        rows.append({"dataset": dataset, "protocol": protocol, "label": label, "metric": metric,
                     "split": split, "n": len(values), "mean": statistics.mean(values),
                     "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                     "values": values, **meta.get(key, {})})
    return sorted(rows, key=lambda row: (row["dataset"], row["protocol"], row["metric"], row["label"], row["split"]))


def primary_metric(label: str, protocol: str) -> str:
    a1_score, a2_score = PRIMARY_SCORE.get(label, ("msp", "msp"))
    return f"{a1_score if protocol == 'A1' else a2_score}.f1"


def write_markdown(rows: list[dict], output: Path) -> None:
    lines = ["# SASeg revision results", ""]
    keys = sorted({(row["dataset"], row["protocol"]) for row in rows})
    for dataset, protocol in keys:
        subset = [row for row in rows if row["dataset"] == dataset and row["protocol"] == protocol
                  and row["split"] == "test"]
        if not subset:
            continue
        labels = sorted({row["label"] for row in subset})
        lines += [f"## {dataset} / {protocol}", "",
                  "| model | known mIoU | unknown F1 | AUROC | AUPR | FPR@TPR95 | component F1 | seeds |",
                  "|---|---|---|---|---|---|---|---|"]
        for label in labels:
            def cell(metric: str, scale: float = 1.0) -> str:
                item = next((row for row in subset if row["label"] == label and row["metric"] == metric), None)
                if item is None:
                    return "---"
                return f"{scale * item['mean']:.3f} ± {scale * item['std']:.3f}"
            primary = primary_metric(label, protocol).rsplit(".", 1)[0]
            n = max((row["n"] for row in subset if row["label"] == label), default=0)
            lines.append(f"| {label} | {cell('known_macro_dice')} | {cell(primary + '.f1')} | "
                         f"{cell(primary + '.auroc')} | {cell(primary + '.aupr')} | "
                         f"{cell(primary + '.fpr_at_tpr95')} | {cell(primary + '.component_f1')} | {n} |")
        lines.append("")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed run directories")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.run_root)
    if not runs:
        raise SystemExit(f"no final/metrics.json under {args.run_root}")
    rows = aggregate(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"runs": len(runs), "rows": rows}, indent=2) + "\n")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "protocol", "label", "variant", "unknown_queries",
                                                    "metric", "split", "n", "mean", "std"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    write_markdown(rows, args.output_dir / "summary.md")
    print(json.dumps({"runs": len(runs), "rows": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
