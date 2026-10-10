"""Spatial anomaly module — weed-aware (WeedsGalore) and stress-aware (rice).

The original architecture used NDVI/NDRE to flag chlorosis / nutrient stress
("yellowing patches"). That signal is wrong for the WeedsGalore crop-vs-weed
task. This module supports four modes:

  weed         ExG + NGRDI  (broadleaf-weed discriminators; default for
                              WeedsGalore)
  stress       NDVI + NDRE   (chlorosis / yellowing; reserved for the rice
                              fine-tune)
  uncertainty  seg-head softmax entropy (data-driven; needs the seg logits
                                          passed in)
  union        OR of every available signal

Default band order matches WeedsGalore MSI:
  0: Blue, 1: Green, 2: Red, 3: Red-Edge, 4: NIR.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Mode = Literal["weed", "stress", "uncertainty", "union"]


def ndvi(ms: np.ndarray, red_idx: int = 2, nir_idx: int = 4, eps: float = 1e-6) -> np.ndarray:
    """ms: (C, H, W). Returns (H, W) NDVI in [-1, 1]."""
    red = ms[red_idx].astype(np.float32)
    nir = ms[nir_idx].astype(np.float32)
    return (nir - red) / (nir + red + eps)


def ndre(ms: np.ndarray, rededge_idx: int = 3, nir_idx: int = 4, eps: float = 1e-6) -> np.ndarray:
    re = ms[rededge_idx].astype(np.float32)
    nir = ms[nir_idx].astype(np.float32)
    return (nir - re) / (nir + re + eps)


def exg(ms: np.ndarray, blue_idx: int = 0, green_idx: int = 1, red_idx: int = 2) -> np.ndarray:
    """Excess Green index (Woebbecke 1995): 2G - R - B. Highlights vegetation."""
    b = ms[blue_idx].astype(np.float32)
    g = ms[green_idx].astype(np.float32)
    r = ms[red_idx].astype(np.float32)
    return 2.0 * g - r - b


def ngrdi(ms: np.ndarray, green_idx: int = 1, red_idx: int = 2, eps: float = 1e-6) -> np.ndarray:
    """Normalized Green-Red Difference: (G - R)/(G + R)."""
    g = ms[green_idx].astype(np.float32)
    r = ms[red_idx].astype(np.float32)
    return (g - r) / (g + r + eps)


def softmax_entropy(logits: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """logits: (C, H, W). Returns per-pixel entropy in nats, shape (H, W)."""
    x = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(x); p = p / p.sum(axis=0, keepdims=True)
    return -(p * np.log(p + eps)).sum(axis=0)


def anomaly_mask(
    ms: np.ndarray,
    mode: Mode = "weed",
    seg_logits: np.ndarray | None = None,
    ndvi_thr: float = 0.3,
    ndre_thr: float = 0.15,
    exg_thr: float = 0.05,
    ngrdi_thr: float = 0.0,
    entropy_thr: float = 1.0,
    min_pixels: int = 64,
) -> np.ndarray:
    """Returns a uint8 (H, W) mask where 1 = flagged anomaly.

    `mode` selects which signals contribute. For `uncertainty` and `union`,
    the caller must pass `seg_logits` of shape (C, H, W) from the seg head.
    """
    from skimage.measure import label, regionprops

    H, W = ms.shape[1], ms.shape[2]
    raws: list[np.ndarray] = []

    if mode in ("weed", "union"):
        # Weeds are the *high* end of ExG/NGRDI relative to the canopy
        # baseline; threshold is a percentile-like cutoff applied above zero.
        e = exg(ms)
        n = ngrdi(ms)
        raws.append((e > exg_thr * e.max()).astype(np.uint8))
        raws.append((n > ngrdi_thr).astype(np.uint8))
    if mode in ("stress", "union"):
        v = ndvi(ms); r = ndre(ms)
        raws.append(((v < ndvi_thr) & (r < ndre_thr)).astype(np.uint8))
    if mode in ("uncertainty", "union"):
        if seg_logits is None:
            if mode == "uncertainty":
                raise ValueError("mode='uncertainty' requires seg_logits")
        else:
            ent = softmax_entropy(seg_logits)
            raws.append((ent > entropy_thr).astype(np.uint8))

    if not raws:
        return np.zeros((H, W), dtype=np.uint8)

    raw = np.zeros((H, W), dtype=np.uint8)
    for r in raws:
        raw |= r

    lbl = label(raw)
    out = np.zeros_like(raw)
    for region in regionprops(lbl):
        if region.area >= min_pixels:
            out[lbl == region.label] = 1
    return out
