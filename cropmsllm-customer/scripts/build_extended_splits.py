"""Build extended train + balanced eval splits to enable Task-1 (Health) F1.

The default WeedsGalore train/val/test splits sample only low-NDVI tiles
(all 156 tiles have mean NDVI < 0.6). The full labeled corpus has 756 tiles -
of which 128 are mean-NDVI &gt;= 0.6 ("Healthy") and 447 are 0.4-0.6 ("moderate").
None of those are in the current splits, so Task-1 Health is single-class on
the test set and F1 is undefined.

This script:
  1. Scans every labeled tile on disk for NDVI mean
  2. Writes splits/train_extended.txt: original 104 + 200 sampled from the
     unused pool (stratified by NDVI bucket so the model sees the full range)
  3. Writes splits/test_balanced.txt: original 26 + 30 healthy + 30 moderate
     unused tiles (mixed NDVI) - used ONLY for Task-1 Health F1.
     The original splits/test.txt stays untouched for WACV-comparable mIoU.
"""
from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
DATA = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"
SPLITS_DIR = DATA / "splits"


def load_band_norm(p: Path) -> np.ndarray:
    arr = np.array(Image.open(p), dtype=np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
    return arr / 255.0


def tile_ndvi_mean(tile_id: str) -> float:
    date = tile_id[:10]
    img_dir = DATA / date / "images"
    nir = load_band_norm(img_dir / f"{tile_id}_NIR.png")
    red = load_band_norm(img_dir / f"{tile_id}_R.png")
    return float(((nir - red) / (nir + red + 1e-6)).mean())


def main():
    random.seed(42)

    # Load existing splits
    in_splits: dict[str, set[str]] = {}
    for s in ("train", "val", "test"):
        p = SPLITS_DIR / f"{s}.txt"
        in_splits[s] = set(ln.strip() for ln in open(p) if ln.strip())
    used = set().union(*in_splits.values())

    # Find all labeled tiles on disk
    all_tiles = []
    for date_dir in sorted(DATA.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "splits": continue
        img_dir = date_dir / "images"
        if not img_dir.is_dir(): continue
        for nir_path in sorted(img_dir.glob("*_NIR.png")):
            tid = nir_path.name.replace("_NIR.png", "")
            all_tiles.append(tid)

    # Stat each tile not in the original splits
    pool: list[tuple[str, float]] = []
    for tid in all_tiles:
        if tid in used: continue
        ndvi = tile_ndvi_mean(tid)
        pool.append((tid, ndvi))

    print(f"[scan] total tiles on disk: {len(all_tiles)}")
    print(f"[scan] in current splits  : {len(used)}")
    print(f"[scan] in unused pool     : {len(pool)}")

    # Bucket the pool
    healthy = sorted([t for t in pool if t[1] >= 0.6], key=lambda x: -x[1])
    moderate = sorted([t for t in pool if 0.4 <= t[1] < 0.6], key=lambda x: -x[1])
    low = sorted([t for t in pool if t[1] < 0.4], key=lambda x: -x[1])
    print(f"[buckets] healthy(&gt;=0.6)={len(healthy)}  moderate(0.4-0.6)={len(moderate)}  low(<0.4)={len(low)}")

    # Build:
    #   train_full     = original 104 + EVERY unused tile (max-data run)
    #   train_extended = original 104 + 185 stratified (smaller, kept for ablation)
    #   test_balanced  = original 26 + 30 healthy + 30 moderate (eval-only, NOT in train)
    random.shuffle(healthy); random.shuffle(moderate); random.shuffle(low)

    # First: pull eval-only tiles OUT of the pool so they never leak into train
    test_extra_healthy_ids = set([t[0] for t in healthy[:30]])
    test_extra_moderate_ids = set([t[0] for t in moderate[:30]])
    test_eval_only = test_extra_healthy_ids | test_extra_moderate_ids

    # train_full: all remaining unused tiles (~540 after pulling 60 for eval)
    train_full_extra = [t[0] for t in pool if t[0] not in test_eval_only]
    train_full = list(in_splits["train"]) + train_full_extra

    # train_extended (smaller stratified mix, for ablation)
    train_extra_healthy = [t[0] for t in healthy[30:30 + 80] if t[0] not in test_eval_only]
    train_extra_moderate = [t[0] for t in moderate[30:30 + 80] if t[0] not in test_eval_only]
    train_extra_low = [t[0] for t in low[:40] if t[0] not in test_eval_only]
    train_extended = list(in_splits["train"]) + train_extra_healthy + train_extra_moderate + train_extra_low

    test_balanced = list(in_splits["test"]) + list(test_extra_healthy_ids) + list(test_extra_moderate_ids)

    (SPLITS_DIR / "train_full.txt").write_text("\n".join(sorted(set(train_full))) + "\n")
    print(f"[write] splits/train_full.txt:    {len(set(train_full))} tiles "
          f"({len(in_splits['train'])} original + {len(train_full_extra)} extra)")

    # Persist
    (SPLITS_DIR / "train_extended.txt").write_text("\n".join(sorted(set(train_extended))) + "\n")
    (SPLITS_DIR / "test_balanced.txt").write_text("\n".join(sorted(set(test_balanced))) + "\n")
    n_ext_extra = len(train_extra_healthy) + len(train_extra_moderate) + len(train_extra_low)
    print(f"[write] splits/train_extended.txt: {len(set(train_extended))} tiles "
          f"({len(in_splits['train'])} original + {n_ext_extra} extra)")
    print(f"[write] splits/test_balanced.txt:  {len(set(test_balanced))} tiles "
          f"({len(in_splits['test'])} original + 60 extra)")
    print()
    print("Train-extended NDVI distribution:")
    bins = {">=0.6": 0, "0.4-0.6": 0, "0.2-0.4": 0, "<0.2": 0}
    for tid in set(train_extended):
        n = tile_ndvi_mean(tid)
        if n >= 0.6: bins[">=0.6"] += 1
        elif n >= 0.4: bins["0.4-0.6"] += 1
        elif n >= 0.2: bins["0.2-0.4"] += 1
        else: bins["<0.2"] += 1
    for k, v in bins.items(): print(f"  {k:<8}: {v}")
    print("Test-balanced NDVI distribution:")
    bins = {">=0.6": 0, "0.4-0.6": 0, "<0.4": 0}
    for tid in set(test_balanced):
        n = tile_ndvi_mean(tid)
        if n >= 0.6: bins[">=0.6"] += 1
        elif n >= 0.4: bins["0.4-0.6"] += 1
        else: bins["<0.4"] += 1
    for k, v in bins.items(): print(f"  {k:<8}: {v}")


if __name__ == "__main__":
    main()
