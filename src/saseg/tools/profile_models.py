from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from saseg.models.factory import VARIANTS, build_model


def profile(variant: str, num_known: int, size: int, device: torch.device, repeats: int,
            unknown_queries: int, backbone: str) -> dict:
    model = build_model(variant, num_known, backbone=backbone, pretrained=False, image_size=size,
                        unknown_queries=unknown_queries).to(device).eval()
    image = torch.randn(1, 3, size, size, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as counter:
            model(image)
        flops = int(counter.get_total_flops())
    except Exception:
        flops = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return {"variant": variant, "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "forward_flops": flops, "forward_ms_median": 1000 * statistics.median(timings),
            "forward_ms_min": 1000 * min(timings), "peak_memory_mb": peak,
            "input_size": size, "repeats": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter, FLOP, latency, and memory profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--num-known", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--unknown-queries", type=int, default=2)
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = [profile(variant, args.num_known, args.size, device, args.repeats,
                    args.unknown_queries, args.backbone)
            for variant in [item for item in args.variants.split(",") if item.strip()]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": str(device), "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
