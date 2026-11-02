from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, channels: tuple[int, ...], dim: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(ch, dim, 1) for ch in channels)
        self.output = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1) for _ in channels)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        normalized_features = []
        for feature, lateral in zip(features, self.lateral):
            if feature.ndim != 4:
                raise ValueError(f"expected 4D feature map, got {tuple(feature.shape)}")
            if feature.shape[1] == lateral.in_channels:
                normalized_features.append(feature)
            elif feature.shape[-1] == lateral.in_channels:
                normalized_features.append(feature.permute(0, 3, 1, 2).contiguous())
            else:
                raise ValueError(f"feature shape {tuple(feature.shape)} does not match {lateral.in_channels} channels")
        features = normalized_features
        pyramid: list[torch.Tensor | None] = [None] * len(features)
        top = None
        for i in reversed(range(len(features))):
            lateral = self.lateral[i](features[i])
            top = lateral if top is None else lateral + F.interpolate(top, size=lateral.shape[-2:], mode="nearest")
            pyramid[i] = self.output[i](top)
        return pyramid[0]  # type: ignore[return-value]


class PerPixelSegmenter(nn.Module):
    """Matched Swin/FPN per-pixel K or K+1 baseline used by the reviewer controls."""

    def __init__(self, num_classes: int, dim: int = 256, backbone: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = False, image_size: int = 224):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(0, 1, 2, 3), img_size=image_size)
        self.pixel_decoder = FPN(tuple(self.backbone.feature_info.channels()), dim)
        self.classifier = nn.Conv2d(dim, num_classes, 1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder_image = image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
        encoder_image = (encoder_image - self.image_mean) / self.image_std
        features = self.pixel_decoder(self.backbone(encoder_image))
        logits = self.classifier(features)
        return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


class BinaryBranchSegmenter(nn.Module):
    """Swin/FPN segmenter with a separate binary unknown head (reviewer control).

    The known head predicts background + K known classes; the unknown head is an
    independent binary pathway.  This tests whether a dedicated unknown branch,
    without competition inside a shared output space, is sufficient.
    """

    def __init__(self, num_known: int, dim: int = 256, backbone: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = False, image_size: int = 224):
        super().__init__()
        import timm

        self.num_known = num_known
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(0, 1, 2, 3), img_size=image_size)
        if hasattr(self.backbone, "layers") and len(self.backbone.layers) >= 2:
            for stage in self.backbone.layers[:2]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(False)
        channels = tuple(self.backbone.feature_info.channels())
        self.pixel_decoder = FPN(channels, dim)
        self.known_head = nn.Conv2d(dim, num_known + 1, 1)
        self.unknown_head = nn.Conv2d(dim, 1, 1)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_image = image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
        encoder_image = (encoder_image - self.image_mean) / self.image_std
        features = self.pixel_decoder(self.backbone(encoder_image))
        known_logits = F.interpolate(self.known_head(features), image.shape[-2:], mode="bilinear", align_corners=False)
        unknown_logits = F.interpolate(self.unknown_head(features), image.shape[-2:], mode="bilinear", align_corners=False)
        unknown_prob = unknown_logits.sigmoid()
        confidence = known_logits.softmax(dim=1).amax(dim=1)
        return {"logits": known_logits, "known_logits": known_logits, "unknown_logits": unknown_logits,
                "unknown_probs": unknown_prob, "unknown_union": unknown_prob[:, 0], "confidence": confidence,
                "confidence_logits": torch.logit(confidence.clamp(1e-4, 1 - 1e-4))}


class MaskedQueryLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.heads = heads

    def forward(self, query: torch.Tensor, memory: torch.Tensor, mask_logits: torch.Tensor | None) -> torch.Tensor:
        query = self.norm1(query + self.self_attention(query, query, query, need_weights=False)[0])
        attn_mask = None
        if mask_logits is not None:
            allowed = mask_logits.sigmoid().flatten(2) >= 0.5
            empty = ~allowed.any(dim=-1)
            if empty.any():
                best = mask_logits.flatten(2).argmax(dim=-1, keepdim=True)
                allowed.scatter_(2, best, True)
            blocked = ~allowed
            attn_mask = blocked[:, None].expand(-1, self.heads, -1, -1).reshape(
                query.shape[0] * self.heads, query.shape[1], memory.shape[1]
            )
        cross = self.cross_attention(query, memory, memory, attn_mask=attn_mask, need_weights=False)[0]
        query = self.norm2(query + cross)
        return self.norm3(query + self.ffn(query))


class MaskedQueryDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(MaskedQueryLayer(dim, heads) for _ in range(layers))

    def forward(self, query: torch.Tensor, memory: torch.Tensor, pixel_features: torch.Tensor,
                mask_embed: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        masks = None
        hidden = query
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, memory, masks if index > 0 else None)
            embeddings = mask_embed(hidden)
            masks = torch.einsum("bqd,bdhw->bqhw", embeddings, pixel_features) / (pixel_features.shape[1] ** 0.5)
        return hidden, masks


class SASeg(nn.Module):
    """Swin/FPN region parser with native unknown slots and prototype support."""

    def __init__(self, num_known: int, dim: int = 256, unknown_queries: int = 2, spheres_per_class: int = 2,
                 decoder_layers: int = 3, backbone: str = "swin_tiny_patch4_window7_224", pretrained: bool = False,
                 image_size: int = 224):
        super().__init__()
        import timm

        if unknown_queries < 0:
            raise ValueError("unknown_queries cannot be negative")
        self.num_known = num_known
        self.num_unknown_queries = unknown_queries
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(0, 1, 2, 3), img_size=image_size)
        if hasattr(self.backbone, "layers") and len(self.backbone.layers) >= 2:
            for stage in self.backbone.layers[:2]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(False)
        channels = tuple(self.backbone.feature_info.channels())
        self.pixel_decoder = FPN(channels, dim)
        self.known_queries = nn.Parameter(torch.randn(num_known, dim) * 0.02)
        self.unknown_queries = nn.Parameter(torch.randn(unknown_queries, dim) * 0.02) if unknown_queries else None
        self.background_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.no_object_query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.query_type = nn.Parameter(torch.randn(num_known + unknown_queries + 2, dim) * 0.02)
        self.decoder = MaskedQueryDecoder(dim, heads=8, layers=decoder_layers)
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.confidence = nn.Linear(dim, 1)
        self.num_spheres = spheres_per_class
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_known, spheres_per_class, dim), dim=-1), requires_grad=False)
        self.radii = nn.Parameter(torch.full((num_known, spheres_per_class), 0.5))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_image = image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
        encoder_image = (encoder_image - self.image_mean) / self.image_std
        features = self.backbone(encoder_image)
        pixels = self.pixel_decoder(features)
        b, d, h, w = pixels.shape
        queries = [self.known_queries]
        if self.unknown_queries is not None:
            queries.append(self.unknown_queries)
        queries.extend([self.background_query, self.no_object_query])
        query = torch.cat(queries, dim=0).unsqueeze(0).expand(b, -1, -1) + self.query_type.unsqueeze(0)
        memory = pixels.flatten(2).transpose(1, 2)
        decoded, low_res_logits = self.decoder(query, memory, pixels, self.mask_embed)
        logits = low_res_logits
        logits = F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)
        confidence_logits = self.confidence(decoded[:, :self.num_known]).squeeze(-1)
        confidence = torch.sigmoid(confidence_logits)
        known_logits = logits[:, :self.num_known]
        unknown_logits = logits[:, self.num_known:self.num_known + self.num_unknown_queries]
        pixel_features = F.interpolate(pixels, image.shape[-2:], mode="bilinear", align_corners=False)
        known_probs = known_logits.sigmoid()
        # Full-resolution probability sums overflow FP16 at 512x512.  Keep the
        # region pooling and hypersphere geometry in FP32 under autocast.
        with torch.autocast(device_type=image.device.type, enabled=False):
            pooled_features = pixel_features.float()
            pooled_known_probs = known_probs.float()
            denom = pooled_known_probs.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            region_embeddings = torch.einsum("bkhw,bdhw->bkd", pooled_known_probs, pooled_features) / denom.flatten(2)
            if self.num_unknown_queries:
                unknown_probs = unknown_logits.sigmoid()
                unknown_union = unknown_probs.max(dim=1).values
                pooled_unknown_probs = unknown_probs.float()
                unknown_denom = pooled_unknown_probs.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                unknown_embeddings = torch.einsum("bmhw,bdhw->bmd", pooled_unknown_probs, pooled_features) / unknown_denom.flatten(2)
            else:
                unknown_probs = logits[:, :0]
                unknown_union = image.new_zeros((b, *image.shape[-2:]))
                unknown_embeddings = pixel_features.new_zeros((b, 0, d))
            normalized_z = F.normalize(region_embeddings, dim=-1)
            normalized_p = F.normalize(self.prototypes.float(), dim=-1)
            distances = torch.linalg.vector_norm(normalized_z[:, :, None, None, :] - normalized_p[None], dim=-1)
            support_score = (distances - self.radii.float().clamp_min(1e-4)[None]).flatten(2).amin(dim=-1)
            unknown_z = F.normalize(unknown_embeddings, dim=-1)
            unknown_distances = torch.linalg.vector_norm(unknown_z[:, :, None, None, :] - normalized_p[None], dim=-1)
            unknown_support_score = (unknown_distances - self.radii.float().clamp_min(1e-4)[None]).flatten(2).amin(dim=-1)
        return {"logits": logits, "known_logits": known_logits, "unknown_logits": unknown_logits,
                "unknown_probs": unknown_probs, "unknown_union": unknown_union, "confidence": confidence,
                "confidence_logits": confidence_logits,
                "region_embeddings": region_embeddings, "unknown_embeddings": unknown_embeddings,
                "support_score": support_score, "unknown_support_score": unknown_support_score, "pixel_features": pixel_features,
                "radii": self.radii, "prototypes": self.prototypes}

    @torch.no_grad()
    def update_prototypes(self, outputs: dict[str, torch.Tensor], target: torch.Tensor,
                          momentum: float = 0.99, min_iou: float = 0.8) -> None:
        masks = outputs["known_logits"].sigmoid() >= 0.5
        features = outputs["region_embeddings"]
        centers = F.normalize(self.prototypes, dim=-1)
        for class_index in range(self.num_known):
            truth = target.eq(class_index + 1)
            for batch_index in range(target.shape[0]):
                if target[batch_index].eq(-100).any():
                    valid = target[batch_index].ne(-100)
                    pred_mask = masks[batch_index, class_index] & valid
                    truth_mask = truth[batch_index] & valid
                else:
                    pred_mask = masks[batch_index, class_index]
                    truth_mask = truth[batch_index]
                union = (pred_mask | truth_mask).sum()
                if union == 0:
                    continue
                iou = (pred_mask & truth_mask).sum().float() / union.float()
                if iou <= min_iou:
                    continue
                vector = F.normalize(features[batch_index, class_index], dim=-1)
                nearest = torch.linalg.vector_norm(centers[class_index] - vector[None], dim=-1).argmin()
                self.prototypes[class_index, nearest].mul_(momentum).add_(vector, alpha=1 - momentum)
                self.prototypes[class_index, nearest].copy_(F.normalize(self.prototypes[class_index, nearest], dim=0))


class ClassAgnosticSASeg(nn.Module):
    """MaskFormer-style class-agnostic queries (reviewer control).

    Query slots carry no class identity.  Every query predicts a mask, and a
    shared linear classifier assigns a class distribution over
    {background, K known, unknown} to the query embedding.  Per-pixel logits are
    the query-mask-weighted sum of those class distributions.  This keeps the
    explicit unknown output class while removing the per-class / native-unknown
    query structure.
    """

    def __init__(self, num_known: int, num_queries: int | None = None, dim: int = 256,
                 decoder_layers: int = 3, backbone: str = "swin_tiny_patch4_window7_224",
                 pretrained: bool = False, image_size: int = 224):
        super().__init__()
        import timm

        self.num_known = num_known
        self.num_queries = num_queries if num_queries is not None else num_known + 4
        if self.num_queries < 1:
            raise ValueError("num_queries must be positive")
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=(0, 1, 2, 3), img_size=image_size)
        if hasattr(self.backbone, "layers") and len(self.backbone.layers) >= 2:
            for stage in self.backbone.layers[:2]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(False)
        channels = tuple(self.backbone.feature_info.channels())
        self.pixel_decoder = FPN(channels, dim)
        self.queries = nn.Parameter(torch.randn(self.num_queries, dim) * 0.02)
        self.decoder = MaskedQueryDecoder(dim, heads=8, layers=decoder_layers)
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.query_classifier = nn.Linear(dim, num_known + 2)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_image = image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image
        encoder_image = (encoder_image - self.image_mean) / self.image_std
        pixels = self.pixel_decoder(self.backbone(encoder_image))
        b, d, h, w = pixels.shape
        query = self.queries.unsqueeze(0).expand(b, -1, -1)
        memory = pixels.flatten(2).transpose(1, 2)
        decoded, low_res_logits = self.decoder(query, memory, pixels, self.mask_embed)
        class_logits = self.query_classifier(decoded)  # B, N, K+2
        mask_weights = low_res_logits.flatten(2).softmax(dim=1)  # B, N, h*w
        pixel_logits = torch.einsum("bnp,bnc->bcp", mask_weights, class_logits).view(b, self.num_known + 2, h, w)
        logits = F.interpolate(pixel_logits, image.shape[-2:], mode="bilinear", align_corners=False)
        known_logits = logits[:, 1:self.num_known + 1]
        unknown_logits = logits[:, self.num_known + 1:self.num_known + 2]
        unknown_probs = unknown_logits.sigmoid()
        confidence = known_logits.sigmoid().amax(dim=1)
        zero = image.new_zeros((b, *image.shape[-2:]))
        return {"logits": logits, "pixel_logits": logits, "known_logits": known_logits,
                "unknown_logits": unknown_logits, "unknown_probs": unknown_probs,
                "unknown_union": unknown_probs[:, 0], "confidence": confidence,
                "confidence_logits": torch.logit(confidence.clamp(1e-4, 1 - 1e-4)),
                "support_score": zero, "unknown_support_score": zero.new_zeros((b, 0)),
                "region_embeddings": decoded.new_zeros((b, 0, d)), "unknown_embeddings": decoded.new_zeros((b, 0, d)),
                "pixel_features": pixels}
