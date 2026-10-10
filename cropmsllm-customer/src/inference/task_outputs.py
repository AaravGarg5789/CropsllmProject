"""Pandiyan's 5-task post-processing layer over the v3/v4 model outputs.

All 5 tasks are derivations from (ms, seg, ndvi_pred, ndre_pred, canopy_scalar).
No new model parameters. Designed to plug into both `scripts/app.py` (UI) and
the LLM prompt builder.

Tasks:
    1. crop_health(ndvi_map)             -> Healthy / Stressed (per-tile binary)
    2. health_intensity_rgb(ndvi_map)    -> RGB-colormapped NDVI map (Task 2)
    3. stress_type(ndvi_map, ndre_map, nir_band)
                                          -> Nitrogen-def / Water-stress / General
    4. field_zoning(ndvi_map)            -> good/poor 2-class mask (Task 4)
    5. spatial_text(ndvi_map, seg, stress_type_map)
                                          -> 3x3 grid description for LLM (Task 5)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Pandiyan's NDVI thresholds (from his message)
NDVI_HEALTHY_THRESHOLD = 0.6      # > 0.6 = healthy
NDVI_STRESSED_THRESHOLD = 0.4     # < 0.4 = stressed
# In-between (0.4..0.6) is "moderate" — we treat it as stressed for binary task
# since Pandiyan's spec didn't include a moderate label.

# Stress-type heuristic thresholds (precision-ag literature)
NDRE_NITROGEN_THRESHOLD = 0.20    # NDRE below this = chlorophyll-poor
NDVI_VEG_PRESENT_THRESHOLD = 0.30 # vegetation visible but stressed
NDVI_WATER_STRESS_THRESHOLD = 0.30  # below = drought-suspect


# Field-zoning thresholds (NDVI-only, for Task 4 "good/poor" call)
NDVI_GOOD_THRESHOLD = 0.55         # >= -> "good area"


@dataclass
class TaskOutputs:
    crop_health: str                       # "Healthy" or "Stressed"
    crop_health_score: float               # mean NDVI of vegetation pixels in [-1, 1]
    health_intensity_rgb: np.ndarray       # (H, W, 3) uint8, NDVI-colormap
    ndvi_summary: dict                     # {min, mean, p10, p50, p90, max}
    stress_type: str                       # "Nitrogen-def" | "Water-stress" | "General-stress" | "None"
    stress_type_map: np.ndarray            # (H, W) uint8 in {0=none, 1=N, 2=water, 3=general}
    stress_type_legend: dict[int, str]
    field_zone_map: np.ndarray             # (H, W) uint8 in {0=poor, 1=good}
    field_zone_summary: dict               # {good_pct, poor_pct}
    spatial_text: str                      # 3x3 grid of NDVI summaries (for LLM)


def _ndvi_colormap(ndvi: np.ndarray) -> np.ndarray:
    """NDVI in [-1, 1] -> RGB heatmap. Standard agronomic colormap:
    red (low) - yellow - green (high)."""
    n = np.clip((ndvi + 1) / 2.0, 0, 1)        # rescale to [0, 1]
    r = np.clip(2 * (1 - n), 0, 1)
    g = np.clip(2 * n, 0, 1)
    b = np.zeros_like(n)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def crop_health(ndvi_map: np.ndarray, seg_mask: np.ndarray | None = None) -> tuple[str, float]:
    """Task 1: tile-level Healthy/Stressed.

    Computes mean NDVI over vegetation pixels (non-background if seg given,
    else all pixels) and applies Pandiyan's threshold."""
    if seg_mask is not None:
        veg = ndvi_map[seg_mask > 0]
        if veg.size == 0:
            veg = ndvi_map
    else:
        veg = ndvi_map
    score = float(np.mean(veg))
    label = "Healthy" if score >= NDVI_HEALTHY_THRESHOLD else "Stressed"
    return label, score


def health_intensity_rgb(ndvi_map: np.ndarray) -> np.ndarray:
    """Task 2: NDVI as a colormapped RGB image."""
    return _ndvi_colormap(ndvi_map)


def ndvi_summary(ndvi_map: np.ndarray) -> dict:
    return {
        "min": float(ndvi_map.min()),
        "p10": float(np.percentile(ndvi_map, 10)),
        "median": float(np.percentile(ndvi_map, 50)),
        "mean": float(ndvi_map.mean()),
        "p90": float(np.percentile(ndvi_map, 90)),
        "max": float(ndvi_map.max()),
    }


def stress_type(ndvi_map: np.ndarray, ndre_map: np.ndarray,
                nir_band: np.ndarray | None = None) -> tuple[str, np.ndarray, dict[int, str]]:
    """Task 3: per-pixel and tile-level stress type.

    Heuristic (precision-ag literature):
      - Nitrogen deficiency : NDRE < 0.2 AND NDVI > 0.3
      - Water stress        : NDVI < 0.3 AND NIR-percentile < 0.5
      - General stress      : NDVI < 0.4 with neither dominant
    """
    H, W = ndvi_map.shape
    out = np.zeros((H, W), dtype=np.uint8)
    n_def = (ndre_map < NDRE_NITROGEN_THRESHOLD) & (ndvi_map > NDVI_VEG_PRESENT_THRESHOLD)
    if nir_band is not None:
        # Normalize NIR to its own per-tile percentile so the threshold is robust
        nir_flat = nir_band.ravel()
        if nir_flat.size > 0:
            nir_p50 = np.percentile(nir_flat, 50)
            water = (ndvi_map < NDVI_WATER_STRESS_THRESHOLD) & (nir_band < nir_p50)
        else:
            water = ndvi_map < NDVI_WATER_STRESS_THRESHOLD
    else:
        water = ndvi_map < NDVI_WATER_STRESS_THRESHOLD
    general = (ndvi_map < 0.4) & ~n_def & ~water

    out[n_def] = 1
    out[water] = 2
    out[general] = 3
    legend = {0: "no stress", 1: "Nitrogen-def", 2: "Water-stress", 3: "General-stress"}

    counts = np.bincount(out.ravel(), minlength=4)
    nonzero = counts[1:].sum()
    if nonzero == 0:
        label = "None"
    else:
        idx = int(np.argmax(counts[1:]) + 1)
        label = legend[idx]
    return label, out, legend


def field_zoning(ndvi_map: np.ndarray, seg_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Task 4: pixel-level zoning. 0=bare, 1=poor, 2=good.

    With seg_mask: bare = (seg == 0); within vegetation, good iff NDVI >= 0.55.
    Without seg_mask: 2-class — good iff NDVI >= 0.55, else poor.
    """
    H, W = ndvi_map.shape
    out = np.ones((H, W), dtype=np.uint8)        # default: poor
    out[ndvi_map >= NDVI_GOOD_THRESHOLD] = 2     # good
    if seg_mask is not None:
        out[seg_mask == 0] = 0                   # bare ground from seg
    counts = np.bincount(out.ravel(), minlength=3)
    total = max(out.size, 1)
    summary = {
        "bare_pct": float(counts[0] / total * 100.0),
        "poor_pct": float(counts[1] / total * 100.0),
        "good_pct": float(counts[2] / total * 100.0),
    }
    return out, summary


def _grid_label(i: int, j: int) -> str:
    rows = ["upper", "center", "lower"]
    cols = ["left", "center", "right"]
    return f"{rows[i]}-{cols[j]}".replace("center-center", "center")


def spatial_text(ndvi_map: np.ndarray,
                 stress_type_map: np.ndarray) -> str:
    """Task 5: 3x3 grid summary of NDVI + dominant stress per region.

    Returns a multi-line string for the LLM prompt.
    """
    H, W = ndvi_map.shape
    h3, w3 = H // 3, W // 3
    legend = {0: "no stress", 1: "Nitrogen-def", 2: "Water-stress", 3: "General-stress"}
    lines = []
    for i in range(3):
        for j in range(3):
            y0, y1 = i * h3, (i + 1) * h3 if i < 2 else H
            x0, x1 = j * w3, (j + 1) * w3 if j < 2 else W
            patch_ndvi = ndvi_map[y0:y1, x0:x1]
            patch_stress = stress_type_map[y0:y1, x0:x1]
            mean_ndvi = float(patch_ndvi.mean())
            counts = np.bincount(patch_stress.ravel(), minlength=4)
            nz = counts[1:].sum()
            if nz == 0:
                stress_lbl = "no stress"
            else:
                stress_lbl = legend[int(np.argmax(counts[1:]) + 1)]
            health = "healthy" if mean_ndvi >= NDVI_HEALTHY_THRESHOLD \
                     else ("moderate" if mean_ndvi >= NDVI_STRESSED_THRESHOLD else "stressed")
            lines.append(f"  {_grid_label(i, j):<14s} NDVI={mean_ndvi:+.2f}  "
                         f"({health}; dominant: {stress_lbl})")
    return "\n".join(lines)


def compute_all(ms: np.ndarray, seg_mask: np.ndarray,
                ndvi_pred: np.ndarray, ndre_pred: np.ndarray) -> TaskOutputs:
    """Compute all 5 task outputs from model predictions + raw bands.

    ms: (5, H, W) at load_ms_tile scale; band order (B, G, R, RE, NIR).
    """
    nir = ms[4]
    health_label, health_score = crop_health(ndvi_pred, seg_mask)
    intensity_rgb = health_intensity_rgb(ndvi_pred)
    summary = ndvi_summary(ndvi_pred)
    s_label, s_map, s_legend = stress_type(ndvi_pred, ndre_pred, nir_band=nir)
    zone_map, zone_summary = field_zoning(ndvi_pred, seg_mask=seg_mask)
    text = spatial_text(ndvi_pred, s_map)
    return TaskOutputs(
        crop_health=health_label,
        crop_health_score=health_score,
        health_intensity_rgb=intensity_rgb,
        ndvi_summary=summary,
        stress_type=s_label,
        stress_type_map=s_map,
        stress_type_legend=s_legend,
        field_zone_map=zone_map,
        field_zone_summary=zone_summary,
        spatial_text=text,
    )


# Class palette for the stress_type_map UI display
STRESS_PALETTE = np.array([
    [ 30,  30,  30],   # 0 no stress      — dark
    [255, 220,  20],   # 1 Nitrogen-def   — yellow
    [ 50, 130, 255],   # 2 Water-stress   — blue
    [200,  50, 200],   # 3 General-stress — magenta
], dtype=np.uint8)


# Field-zone palette (3-class with bare ground)
ZONE_PALETTE = np.array([
    [120, 100,  80],   # 0 bare ground — sandy
    [200,  60,  40],   # 1 poor        — red
    [ 50, 200,  60],   # 2 good        — green
], dtype=np.uint8)


def colorize_stress(stress_map: np.ndarray) -> np.ndarray:
    return STRESS_PALETTE[np.clip(stress_map, 0, len(STRESS_PALETTE) - 1)]


def colorize_zones(zone_map: np.ndarray) -> np.ndarray:
    return ZONE_PALETTE[np.clip(zone_map, 0, len(ZONE_PALETTE) - 1)]
