"""Test-time augmentation: 4-flip averaged forward.

For each input we run identity + h-flip + v-flip + 180°-rotation, undo each
spatial transform on the seg-logits side, and average. Aux heads (NDVI, NDRE,
canopy) are also averaged where applicable. 1-4 pp mIoU on small UAV
datasets is typical and well-documented.
"""
from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def tta4_predict(model, x: Tensor) -> dict[str, Tensor]:
    """x: (B, C, H, W). Returns averaged outputs (same dict as model(x))."""
    # transform pairs: (forward op, inverse op on logits/dense maps)
    ops = [
        (lambda t: t,                                   lambda t: t),
        (lambda t: torch.flip(t, dims=(-1,)),           lambda t: torch.flip(t, dims=(-1,))),
        (lambda t: torch.flip(t, dims=(-2,)),           lambda t: torch.flip(t, dims=(-2,))),
        (lambda t: torch.flip(t, dims=(-2, -1)),        lambda t: torch.flip(t, dims=(-2, -1))),
    ]
    seg_acc = None; ndvi_acc = None; ndre_acc = None; canopy_acc = None
    n = len(ops)
    for fwd, inv in ops:
        out = model(fwd(x))
        seg = inv(out["seg"])
        ndvi = inv(out["ndvi"])
        ndre = inv(out["ndre"])
        canopy = out["canopy"]   # scalar; flips don't change it
        seg_acc = seg if seg_acc is None else seg_acc + seg
        ndvi_acc = ndvi if ndvi_acc is None else ndvi_acc + ndvi
        ndre_acc = ndre if ndre_acc is None else ndre_acc + ndre
        canopy_acc = canopy if canopy_acc is None else canopy_acc + canopy
    return {"seg": seg_acc / n, "ndvi": ndvi_acc / n,
            "ndre": ndre_acc / n, "canopy": canopy_acc / n}
