from __future__ import annotations

from torch import nn

from saseg.models.saseg import BinaryBranchSegmenter, ClassAgnosticSASeg, PerPixelSegmenter, SASeg

VARIANTS = ("perpixel_k", "perpixel_k1", "saseg", "saseg_generic", "saseg_classagnostic", "binary_branch")


def build_model(variant: str, num_known: int, backbone: str = "swin_tiny_patch4_window7_224",
                pretrained: bool = False, image_size: int = 224, unknown_queries: int = 2,
                spheres_per_class: int = 2, decoder_layers: int = 3) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant in {"saseg", "saseg_generic"}:
        return SASeg(num_known, pretrained=pretrained, image_size=image_size, backbone=backbone,
                     unknown_queries=unknown_queries, spheres_per_class=spheres_per_class,
                     decoder_layers=decoder_layers)
    if variant == "saseg_classagnostic":
        return ClassAgnosticSASeg(num_known, num_queries=num_known + unknown_queries + 2,
                                  pretrained=pretrained, image_size=image_size, backbone=backbone,
                                  decoder_layers=decoder_layers)
    if variant == "binary_branch":
        return BinaryBranchSegmenter(num_known, pretrained=pretrained, image_size=image_size, backbone=backbone)
    return PerPixelSegmenter(num_known + (2 if variant == "perpixel_k1" else 1),
                             pretrained=pretrained, image_size=image_size, backbone=backbone)
