"""v3 multi-task model: SMP DeepLabv3+ ResNet50 + 4 heads at native 600x600.

Heads (same supervision as v2):
  seg            : (B, num_classes, H, W)
  ndvi_dense     : (B, 1, H, W)              tanh-squashed to [-1, 1]
  ndre_dense     : (B, 1, H, W)              tanh-squashed to [-1, 1]
  canopy_scalar  : (B, 1)                    sigmoid in [0, 1]

The seg + dense aux heads share the DeepLabv3+ decoder; the canopy head taps
the deepest encoder feature directly (separate path keeps the global pool
honest).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbone_smp import (decoder_out_channels,
                                      encoder_deepest_channels,
                                      make_smp_deeplab)


class MultiHeadV3(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 5,
                 encoder_name: str = "resnet50") -> None:
        super().__init__()
        self.seg_model = make_smp_deeplab(num_classes=num_classes,
                                          in_channels=in_channels,
                                          encoder_name=encoder_name)
        self.encoder = self.seg_model.encoder
        self.decoder = self.seg_model.decoder
        self.seg_head = self.seg_model.segmentation_head

        d_dec = decoder_out_channels(self.seg_model)   # 256 for DeepLabv3+
        d_enc = encoder_deepest_channels(self.seg_model)  # 2048 for ResNet50

        self.ndvi_head = nn.Sequential(
            nn.Conv2d(d_dec, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        self.ndre_head = nn.Sequential(
            nn.Conv2d(d_dec, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )
        self.canopy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(d_enc, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feats = self.encoder(x)
        deep = feats[-1]
        dec = self.decoder(feats)               # (B, 256, H/2, W/2) at OS=8
        seg_logits = self.seg_head(dec)         # SMP upsamples to (B, C, H, W)

        # NDVI/NDRE: take the decoder feature map, run a small conv head, then
        # bilinearly upsample to (H, W) so it lines up with input + supervision.
        H, W = x.shape[-2], x.shape[-1]
        ndvi = torch.tanh(self.ndvi_head(dec))
        ndre = torch.tanh(self.ndre_head(dec))
        if ndvi.shape[-2:] != (H, W):
            import torch.nn.functional as F
            ndvi = F.interpolate(ndvi, size=(H, W), mode="bilinear", align_corners=False)
            ndre = F.interpolate(ndre, size=(H, W), mode="bilinear", align_corners=False)

        canopy = torch.sigmoid(self.canopy_head(deep))
        return {"seg": seg_logits, "ndvi": ndvi, "ndre": ndre, "canopy": canopy}

    def freeze_encoder(self, freeze: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
