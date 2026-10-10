"""v2 multi-task head over a frozen-or-LoRA shared backbone.

Heads:
    seg            : (B, 6, H, W)  — primary, 6-class WeedsGalore mask
    ndvi_dense     : (B, 1, H, W)  — pixel NDVI = (NIR-R)/(NIR+R), in [-1, 1]
    ndre_dense     : (B, 1, H, W)  — pixel NDRE = (NIR-RE)/(NIR+RE), in [-1, 1]
    canopy_scalar  : (B, 1)        — per-tile canopy fraction = 1 - bg ratio

Decoder is a lightweight FPN-ish head over the (B, 768, 14, 14) DOFA features.
For 600x600 inputs we resize to 224x224 in the data pipeline (DOFA's pretrain
size) and bilinearly upsample logits back to (H, W) at the loss boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )


class SharedDecoder(nn.Module):
    """768-dim ViT tokens → 256-channel feature map at 56×56 (4× upsample)."""

    def __init__(self, in_dim: int = 768, mid: int = 256) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_dim, mid, k=1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(mid, mid, 2, stride=2), ConvBNAct(mid, mid))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(mid, mid, 2, stride=2), ConvBNAct(mid, mid))

    def forward(self, x: Tensor) -> Tensor:  # (B, 768, 14, 14) -> (B, 256, 56, 56)
        return self.up2(self.up1(self.proj(x)))


class MultiHeadModel(nn.Module):
    """DOFA backbone + shared decoder + 4 heads."""

    def __init__(self, backbone: nn.Module, num_classes: int = 6, mid: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = SharedDecoder(in_dim=backbone.embed_dim, mid=mid)
        self.seg_head = nn.Conv2d(mid, num_classes, 1)
        self.ndvi_head = nn.Conv2d(mid, 1, 1)
        self.ndre_head = nn.Conv2d(mid, 1, 1)
        self.canopy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x: Tensor, output_size: tuple[int, int] | None = None) -> dict[str, Tensor]:
        feats = self.backbone(x)             # (B, 768, 14, 14)
        d = self.decoder(feats)              # (B, 256, 56, 56)
        seg_logits = self.seg_head(d)
        ndvi = torch.tanh(self.ndvi_head(d))   # squash to [-1, 1]
        ndre = torch.tanh(self.ndre_head(d))
        canopy = torch.sigmoid(self.canopy_head(d))  # [0, 1]
        if output_size is not None:
            seg_logits = F.interpolate(seg_logits, size=output_size, mode="bilinear", align_corners=False)
            ndvi = F.interpolate(ndvi, size=output_size, mode="bilinear", align_corners=False)
            ndre = F.interpolate(ndre, size=output_size, mode="bilinear", align_corners=False)
        return {"seg": seg_logits, "ndvi": ndvi, "ndre": ndre, "canopy": canopy}


def compute_supervision_targets(ms: Tensor, mask: Tensor, eps: float = 1e-6) -> dict[str, Tensor]:
    """Derive NDVI/NDRE/canopy targets from MS bands + seg mask.

    ms: (B, 5, H, W) in some scale (load_ms_tile divides 16-bit / 255 → ~0-256).
        Spectral indices are scale-invariant ratios so we don't need to renormalize.
    mask: (B, H, W) class indices.
    """
    blue, green, red, rededge, nir = ms[:, 0], ms[:, 1], ms[:, 2], ms[:, 3], ms[:, 4]
    ndvi = (nir - red) / (nir + red + eps)
    ndre = (nir - rededge) / (nir + rededge + eps)
    canopy = 1.0 - (mask == 0).float().mean(dim=(-1, -2))  # (B,)
    return {
        "ndvi": ndvi.unsqueeze(1).clamp(-1, 1),    # (B, 1, H, W)
        "ndre": ndre.unsqueeze(1).clamp(-1, 1),
        "canopy": canopy.unsqueeze(1),              # (B, 1)
    }
