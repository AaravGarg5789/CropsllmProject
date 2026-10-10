"""Sliding-window inference with overlap + Gaussian-weighted blending.

For a 600×600 input, predicts on 4 overlapping 384×384 crops (top-left,
top-right, bot-left, bot-right) and blends logits with a 2D Gaussian weight
per crop. Combined with 4-flip TTA per crop. Robust on small-instance seg
where edge pixels in any single forward pass have weak context.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.inference.tta import tta4_predict


def _gaussian_2d(h: int, w: int, sigma_frac: float = 0.2, device=None) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    sigma = sigma_frac
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g  # (H, W)


@torch.no_grad()
def sliding_window_predict(model, x: Tensor, crop: int = 384,
                           stride: int = 216, use_tta: bool = True) -> dict[str, Tensor]:
    """x: (B, C, H, W). Returns blended outputs at (H, W).

    crop=384 with stride=216 covers a 600×600 with 4 crops:
        (0..384, 0..384), (216..600, 0..384),
        (0..384, 216..600), (216..600, 216..600)
    """
    B, C, H, W = x.shape
    device = x.device
    seg_acc = torch.zeros(B, model.seg_head[0].out_channels if hasattr(model, "seg_head") else 6,
                          H, W, device=device)
    ndvi_acc = torch.zeros(B, 1, H, W, device=device)
    ndre_acc = torch.zeros(B, 1, H, W, device=device)
    weight = torch.zeros(B, 1, H, W, device=device)
    canopy_sum = torch.zeros(B, 1, device=device); n_canopy = 0

    coords = []
    for y in range(0, max(1, H - crop + 1), stride):
        for x_ in range(0, max(1, W - crop + 1), stride):
            coords.append((y, x_))
        # ensure right edge covered
        if (W - crop) > 0 and (W - crop) % stride != 0:
            coords.append((y, W - crop))
    if (H - crop) > 0 and (H - crop) % stride != 0:
        for x_ in range(0, max(1, W - crop + 1), stride):
            coords.append((H - crop, x_))
        if (W - crop) > 0 and (W - crop) % stride != 0:
            coords.append((H - crop, W - crop))
    coords = list({c for c in coords})  # dedup

    g = _gaussian_2d(crop, crop, sigma_frac=0.25, device=device).unsqueeze(0).unsqueeze(0)

    for (y, x_) in coords:
        patch = x[:, :, y:y + crop, x_:x_ + crop]
        if use_tta:
            out = tta4_predict(model, patch)
        else:
            out = model(patch)
        seg_acc[:, :, y:y + crop, x_:x_ + crop] += out["seg"] * g
        ndvi_acc[:, :, y:y + crop, x_:x_ + crop] += out["ndvi"] * g
        ndre_acc[:, :, y:y + crop, x_:x_ + crop] += out["ndre"] * g
        weight[:, :, y:y + crop, x_:x_ + crop] += g
        canopy_sum += out["canopy"]; n_canopy += 1

    seg = seg_acc / weight.clamp(min=1e-6)
    ndvi = ndvi_acc / weight.clamp(min=1e-6)
    ndre = ndre_acc / weight.clamp(min=1e-6)
    canopy = canopy_sum / max(n_canopy, 1)
    return {"seg": seg, "ndvi": ndvi, "ndre": ndre, "canopy": canopy}
