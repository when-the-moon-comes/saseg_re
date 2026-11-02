from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from saseg.data.labels import LabelMapping
from saseg.data.nifti import NiftiSliceDataset, build_slice_records
from saseg.losses import saseg_loss
from saseg.models.saseg import SASeg


class SyntheticSlices(Dataset):
    """Small deterministic smoke-test set; never used for paper results."""

    def __init__(self, count: int, known: int, size: int, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(count, 1, size, size, generator=generator)
        self.targets = torch.zeros(count, size, size, dtype=torch.long)
        for index in range(count):
            self.targets[index, 8:24, 8:24] = 1
            if known > 1:
                self.targets[index, 32:48, 32:48] = 2
            self.targets[index, 18:22, 34:38] = known + 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "target": self.targets[index]}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="SASeg smoke training entry point")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", choices=["A1", "A2"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-known", type=int)
    parser.add_argument("--unknown-queries", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(args.config.read_text()) if args.config else {}
    def value(name: str, default: Any) -> Any:
        cli_value = getattr(args, name, None)
        return cli_value if cli_value is not None else config.get(name.replace("-", "_"), default)

    seed = int(value("seed", 42))
    seed_everything(seed)
    device = torch.device(value("device", "cuda" if torch.cuda.is_available() else "cpu"))
    protocol = value("protocol", "A1")
    size = int(value("size", 512))
    batch_size = int(value("batch_size", 8))
    max_steps = value("max_steps", None)
    epochs = int(value("epochs", 100))
    unknown_queries = int(value("unknown_queries", 2))
    output_dir = Path(value("output_dir", "outputs/checkpoints/smoke"))

    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        label_cfg = manifest["labels"]
        mapping = LabelMapping(tuple(label_cfg["known"]), tuple(label_cfg["seen_unknown"]),
                               tuple(label_cfg["unseen_unknown"]), int(label_cfg.get("background", 0)))
        records = [record for record in build_slice_records(args.manifest) if record["split"] == "train"]
        if not records:
            raise ValueError("manifest has no training slices; add explicit slice_indices for train cases")
        dataset: Dataset = NiftiSliceDataset(records, mapping, protocol, role="train", size=size)
        num_known = mapping.num_known
    else:
        num_known = int(value("num_known", 2))
        dataset = SyntheticSlices(max(int(max_steps or 10), batch_size * 2), num_known, size, seed)

    model = SASeg(num_known, dim=256, unknown_queries=unknown_queries,
                  spheres_per_class=int(value("spheres_per_class", 2)),
                  pretrained=bool(value("pretrained", False))).to(device)
    base_lr = float(value("lr", 1e-4))
    backbone_lr = float(value("backbone_lr", 1e-5))
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad], "lr": base_lr},
    ], weight_decay=float(value("weight_decay", 1e-4)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(value("num_workers", 4)), pin_memory=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    for epoch in range(epochs):
        for batch in loader:
            if max_steps is not None and step >= int(max_steps):
                break
            image, target = batch["image"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)
            outputs = model(image)
            losses = saseg_loss(outputs, target, num_known)
            optimizer.zero_grad(set_to_none=True); losses["total"].backward(); optimizer.step()
            model.update_prototypes(outputs, target)
            record = {"epoch": epoch, "step": step, **{name: float(item.detach().cpu()) for name, item in losses.items()}}
            print(json.dumps(record, sort_keys=True), flush=True)
            step += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "step": step, "seed": seed, "protocol": protocol}, output_dir / "last.pt")
        if max_steps is not None and step >= int(max_steps):
            break


if __name__ == "__main__":
    main()
