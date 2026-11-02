from __future__ import annotations

import argparse
import datetime
import glob
import json
import statistics
from pathlib import Path

LABELS = ["perpixel_k", "perpixel_k1", "saseg", "saseg_m0", "saseg_generic",
          "saseg_classagnostic", "binary_branch"]
SEEDS = [42, 43, 44]
PROTOCOLS = ["A1", "A2"]
PRIMARY = {
    "perpixel_k": ("msp", "msp"),
    "perpixel_k1": ("unknown_class", "unknown_class"),
    "saseg": ("unknown_union", "ensemble"),
    "saseg_m0": ("msp", "msp"),
    "saseg_generic": ("msp", "msp"),
    "saseg_classagnostic": ("unknown_class", "unknown_class"),
    "binary_branch": ("unknown_branch", "unknown_branch"),
}


def fmt(values: list[float | None]) -> str:
    values = [value for value in values if value is not None]
    if not values:
        return "---"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.pstdev(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the running experiment log")
    parser.add_argument("--root", type=Path, required=True, help="e.g. runs/formal_eval/synapse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="Synapse")
    parser.add_argument("--code-md5", default="")
    parser.add_argument("--pending", default="AMOS22 训练、VOC2012/Cityscapes")
    args = parser.parse_args()

    rows: dict[tuple[str, str], list[tuple]] = {}
    done: set[tuple[str, str, int]] = set()
    for path in glob.glob(str(args.root / "*" / "*" / "*" / "final" / "metrics.json")):
        parts = Path(path).parts
        protocol, label = parts[-5], parts[-4]
        seed = int(parts[-3].replace("seed", ""))
        done.add((label, protocol, seed))
        metrics = json.loads(Path(path).read_text())
        known = metrics.get("known_segmentation", {})
        known_dice = (known.get("test") or {}).get("macro_dice")
        score = PRIMARY.get(label, ("msp", "msp"))[0 if protocol == "A1" else 1]
        test = (metrics.get(score) or {}).get("test") or {}
        rows.setdefault((label, protocol), []).append(
            (seed, known_dice, test.get("auroc"), test.get("aupr"), test.get("f1"), test.get("iou")))

    lines = ["# SASeg 修订实验记录", "",
             f"> 自动生成：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
             f"> 代码 md5：`{args.code_md5 or 'n/a'}`  ",
             f"> 数据集：{args.dataset}  ",
             "> 指标：AUROC/AUPR/FPR@TPR95 仅前景像素；F1/IoU 为全像素；本批 eval 未计算 component 指标。", ""]
    total = len(LABELS) * len(PROTOCOLS) * len(SEEDS)
    for protocol in PROTOCOLS:
        lines += [f"## {args.dataset} {protocol}", "",
                  "| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU | seeds |",
                  "|---|---|---|---|---|---|---|---|"]
        for label in LABELS:
            items = sorted(rows.get((label, protocol), []))
            if not items:
                lines.append(f"| {label} | 待跑 | --- | --- | --- | --- | --- | 0/3 |")
                continue
            score = PRIMARY.get(label, ("msp", "msp"))[0 if protocol == "A1" else 1]
            lines.append("| {} | {} | `{}` | {} | {} | {} | {} | {}/3 |".format(
                label, fmt([item[1] for item in items]), score,
                fmt([item[2] for item in items]), fmt([item[3] for item in items]),
                fmt([item[4] for item in items]), fmt([item[5] for item in items]), len(items)))
        lines.append("")

    missing = [f"{label}/{protocol}/seed{seed}" for label in LABELS for protocol in PROTOCOLS
               for seed in SEEDS if (label, protocol, seed) not in done]
    lines += ["## 进度", "", f"- 已完成 eval：**{len(done)} / {total}**"]
    if missing:
        lines.append("- 待补：" + "，".join(missing))
    lines += [f"- 其他待跑：{args.pending}", ""]
    lines += ["## 备注", "",
              "- 本批 eval 复用各 A1 `best.pt`，没有重新训练。",
              "- `saseg` known 预测已修正：未监督 background query 不再参与 argmax；known 类 sigmoid>0.5，否则背景。",
              "- 旧的 `runs/formal/synapse/*/final/metrics.json` 使用修复前指标，不应写入论文。",
              "- `saseg_generic` 的 `objectness` 不是 unknown 分数，不参与主表。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
