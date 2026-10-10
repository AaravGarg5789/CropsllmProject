"""Write a georeferenced prescription GeoTIFF.

Combines the growth-stage segmentation, the <SEG>-conditioned mask from the
LLM, and the spectral anomaly mask into a single multi-band raster with the
input tile's CRS and affine transform preserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_prescription(
    out_path: str | Path,
    growth_stage: np.ndarray,        # (H, W) int
    llm_seg: np.ndarray | None,      # (H, W) uint8 or None
    anomaly: np.ndarray,             # (H, W) uint8
    crs: str,
    transform,                        # rasterio Affine or 6-tuple
) -> None:
    import rasterio
    from rasterio.transform import Affine

    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6])

    bands = [growth_stage.astype(np.uint8), anomaly.astype(np.uint8)]
    # Band 1 carries the segmentation class mask. Name reflects the trained
    # task — "seg_classes" for WeedsGalore (crop/weed) and the rice path will
    # rename to "growth_stage" if/when fine-tuned on rice.
    band_names = ["seg_classes", "anomaly"]
    if llm_seg is not None:
        bands.append(llm_seg.astype(np.uint8))
        band_names.append("llm_seg")

    arr = np.stack(bands, axis=0)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(arr)
        for i, name in enumerate(band_names, start=1):
            ds.set_band_description(i, name)
