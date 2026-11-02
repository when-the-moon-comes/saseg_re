from __future__ import annotations

import torch
from torch.nn import functional as F


def dice_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    target = target.float()
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    valid = valid.expand_as(target)
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1)
    probs = logits.sigmoid()
    dims = tuple(range(1, probs.ndim))
    probs = probs * valid
    target = target * valid
    dice = 1 - ((2 * (probs * target).sum(dims) + eps) / (probs.sum(dims) + target.sum(dims) + eps)).mean()
    return dice + bce


def support_ball_loss(score: torch.Tensor, unknown_score: torch.Tensor, known_valid: torch.Tensor, unknown_valid: torch.Tensor,
                      radii: torch.Tensor, margin: float = 0.5, radius_weight: float = 0.01) -> torch.Tensor:
    pull = F.relu(score[known_valid]).mean() if known_valid.any() else score.new_zeros(())
    # With M=0 unknown queries the support tensor has an empty query axis; a
    # reduction over it would yield NaN and poison the whole loss.
    has_unknown = unknown_score.numel() > 0 and unknown_score.shape[-1] > 0
    push = F.relu(margin - unknown_score[unknown_valid]).mean() if unknown_valid.any() and has_unknown else score.new_zeros(())
    return pull + push + radius_weight * radii.sum()


def boundary_mask(target: torch.Tensor, width: int = 5, sigma: float = 2.0) -> torch.Tensor:
    """Soft boundary around all semantic transitions, excluding image exterior."""
    target = target.float().unsqueeze(1)
    edge = torch.zeros_like(target)
    edge[..., 1:, :] = torch.maximum(edge[..., 1:, :], (target[..., 1:, :] != target[..., :-1, :]).float())
    edge[..., :-1, :] = torch.maximum(edge[..., :-1, :], (target[..., 1:, :] != target[..., :-1, :]).float())
    edge[..., :, 1:] = torch.maximum(edge[..., :, 1:], (target[..., :, 1:] != target[..., :, :-1]).float())
    edge[..., :, :-1] = torch.maximum(edge[..., :, :-1], (target[..., :, 1:] != target[..., :, :-1]).float())
    radius = max(1, width // 2)
    edge = F.max_pool2d(edge, kernel_size=2 * radius + 1, stride=1, padding=radius)
    k = max(3, int(6 * sigma + 1) | 1)
    coords = torch.arange(k, device=target.device, dtype=target.dtype) - k // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel2d = torch.outer(kernel, kernel).view(1, 1, k, k)
    return F.conv2d(edge, kernel2d, padding=k // 2).clamp(0, 1)


def saseg_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, num_known: int,
               lambda_unknown: float = 1.0, lambda_ball: float = 0.5, lambda_boundary: float = 0.3,
               diversity_weight: float = 0.001, generic_queries: bool = False) -> dict[str, torch.Tensor]:
    known_target = torch.stack([target.eq(i + 1) for i in range(num_known)], dim=1).float()
    valid_pixels = target.ne(-100).unsqueeze(1)
    known_loss = dice_bce(outputs["known_logits"], known_target, valid_pixels)
    present = known_target.flatten(2).any(-1).float()
    confidence_loss = F.binary_cross_entropy_with_logits(outputs["confidence_logits"], present)
    known_loss = known_loss + confidence_loss
    if generic_queries:
        # Extra query slots are supervised as generic foreground/objectness
        # queries; they receive no unknown-specific semantics.
        unknown_target = target.ge(1) & target.le(num_known)
        unknown_valid = valid_pixels[:, 0] & target.ne(num_known + 1)
    else:
        unknown_target = target.eq(num_known + 1)
        unknown_valid = valid_pixels[:, 0]
    unknown_target = unknown_target.float()
    if outputs["unknown_logits"].shape[1]:
        union_logits = outputs["unknown_logits"].max(dim=1).values
        union = union_logits.sigmoid()
        valid = unknown_valid.float()
        unknown_bce = (F.binary_cross_entropy_with_logits(union_logits, unknown_target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
        dims = tuple(range(1, union.ndim))
        masked_union = union * valid
        masked_target = unknown_target * valid
        unknown_dice = 1 - ((2 * (masked_union * masked_target).sum(dims) + 1e-6) /
                            (masked_union.sum(dims) + masked_target.sum(dims) + 1e-6)).mean()
        unknown_loss = unknown_bce + unknown_dice
        inter_query = outputs["unknown_probs"][:, 0] * outputs["unknown_probs"][:, 1] if outputs["unknown_probs"].shape[1] > 1 else None
        if inter_query is not None:
            denom = (outputs["unknown_probs"][:, 0] + outputs["unknown_probs"][:, 1] - inter_query).sum().clamp_min(1e-6)
            diversity = F.relu(inter_query.sum() / denom - 0.5)
        else:
            diversity = unknown_loss.new_zeros(())
        unknown_loss = unknown_loss + diversity_weight * diversity
    else:
        unknown_loss = outputs["known_logits"].new_zeros(())
    valid_known = present.bool()
    valid_unknown = unknown_target.flatten(1).any(-1)
    if generic_queries:
        valid_unknown = torch.zeros_like(valid_unknown)
    ball_loss = support_ball_loss(outputs["support_score"], outputs["unknown_support_score"], valid_known, valid_unknown,
                                   outputs.get("radii", outputs["support_score"].new_zeros(1)))
    safe_target = target.masked_fill(target.eq(-100), 0)
    boundary = boundary_mask(safe_target)
    valid_map = target.ne(-100).float()
    boundary_loss = (boundary[:, 0] * outputs["unknown_union"] * valid_map).sum() / valid_map.sum().clamp_min(1)
    if generic_queries:
        boundary_loss = boundary_loss.new_zeros(())
    total = known_loss + lambda_unknown * unknown_loss + lambda_ball * ball_loss + lambda_boundary * boundary_loss
    return {"total": total, "known": known_loss, "unknown": unknown_loss, "ball": ball_loss, "boundary": boundary_loss}


def binary_branch_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, num_known: int,
                       lambda_unknown: float = 1.0) -> dict[str, torch.Tensor]:
    """Known-class CE (unknown pixels ignored) plus a separate binary unknown head."""
    known_target = target.clone()
    known_target[known_target == num_known + 1] = -100
    valid_known = known_target.ne(-100)
    ce = F.cross_entropy(outputs["known_logits"], known_target, ignore_index=-100)
    probs = outputs["known_logits"].softmax(dim=1)
    dice_terms = []
    for class_index in range(1, num_known + 1):
        truth = known_target.eq(class_index) & valid_known
        pred = probs[:, class_index] * valid_known
        numerator = 2 * (pred * truth).sum(dim=(-2, -1)) + 1e-6
        denominator = pred.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1e-6
        dice_terms.append(1 - (numerator / denominator).mean())
    known_loss = ce + (torch.stack(dice_terms).mean() if dice_terms else ce.new_zeros(()))
    unknown_target = target.eq(num_known + 1).float()
    valid = target.ne(-100).float()
    logits = outputs["unknown_logits"][:, 0]
    bce = (F.binary_cross_entropy_with_logits(logits, unknown_target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)
    prob = logits.sigmoid()
    dims = tuple(range(1, prob.ndim))
    masked_prob = prob * valid
    masked_target = unknown_target * valid
    dice = 1 - ((2 * (masked_prob * masked_target).sum(dims) + 1e-6) /
                (masked_prob.sum(dims) + masked_target.sum(dims) + 1e-6)).mean()
    unknown_loss = bce + dice
    total = known_loss + lambda_unknown * unknown_loss
    return {"total": total, "known": known_loss, "unknown": unknown_loss}


def class_agnostic_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, num_known: int,
                        lambda_unknown: float = 1.0) -> dict[str, torch.Tensor]:
    """Per-pixel CE + Dice over {background, K known, unknown} on aggregated mask logits."""
    logits = outputs["pixel_logits"]
    ce = F.cross_entropy(logits, target, ignore_index=-100)
    probs = logits.softmax(dim=1)
    valid = target.ne(-100)
    dice_terms = []
    for class_index in range(num_known + 2):
        truth = target.eq(class_index) & valid
        pred = probs[:, class_index] * valid
        numerator = 2 * (pred * truth).sum(dim=(-2, -1)) + 1e-6
        denominator = pred.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1e-6
        dice_terms.append(1 - (numerator / denominator).mean())
    dice = torch.stack(dice_terms).mean() if dice_terms else ce.new_zeros(())
    total = ce + dice
    return {"total": total, "known": ce, "unknown": dice}
