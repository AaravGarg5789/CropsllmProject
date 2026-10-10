"""Filter the 540 unused-pool extras by label-quality agreement with the strong
v4-ViT-v2 checkpoint (cropmsllm_vit.pt, 81.40% canonical test 3-cls mIoU).

Strategy:
  1. For each extra tile, run v4-ViT-v2 on its MS bands -> 3-class prediction.
  2. Compare prediction to the on-disk semantic mask (merged to 3-class).
  3. Keep tiles whose pixel agreement >= TAU.
  4. Stratify by NDVI bucket so the curated set covers the full health range.

Output: data/.../splits/train_curated.txt = 104 canonical + curated extras.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.inference.tta import tta4_predict
from src.models.multi_head_v3 import MultiHeadV3


DATA = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"
SPLITS = DATA / "splits"
CKPT = REPO / "runs" / "cropmsllm_vit.pt"
SCORES_JSON = REPO / "runs" / "curate_scores.json"   # per-tile (agr, ndvi) cache; reused if present
# Top-K within each NDVI bucket by 3-class agreement. The absolute agreement is noisy
# (strong model itself is OOD on orthomosaic extras), so we use relative ranking
# rather than an absolute tau gate.
PER_BUCKET_TARGETS = {"healthy": 50, "moderate": 50, "low": 15}


def load_band(p: Path) -> np.ndarray:
    arr = np.array(Image.open(p), dtype=np.float32)
    if arr.ndim == 3: arr = arr[..., 0]
    return arr / 255.0


def load_tile(tid: str) -> tuple[np.ndarray, np.ndarray]:
    date = tid[:10]
    bands = [load_band(DATA / date / "images" / f"{tid}_{s}.png")
              for s in ("B", "G", "R", "RE", "NIR")]
    ms = np.stack(bands, axis=0).astype(np.float32) * 255.0  # restore 0..255 scale to match trainer
    mask = np.array(Image.open(DATA / date / "semantics" / f"{tid}.png"), dtype=np.int64)
    return ms, mask


def to_native_600(arr_2d_or_3d: np.ndarray) -> np.ndarray:
    """Upsample 224x224 → 600x600 (orthomosaic tiles), pass-through 600x600."""
    if arr_2d_or_3d.shape[-2:] == (600, 600):
        return arr_2d_or_3d
    t = torch.from_numpy(arr_2d_or_3d).float()
    if t.dim() == 3:                 # (C, H, W)
        t = t.unsqueeze(0)
        t = F.interpolate(t, size=(600, 600), mode="bilinear", align_corners=False).squeeze(0)
    else:                            # (H, W) - treat as nearest for masks
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(600, 600), mode="nearest").squeeze().long()
    return t.numpy() if not isinstance(t, np.ndarray) else t


def main():
    # Read existing splits to know what's "extras" (in train_full but not in canonical train)
    canonical_train = set((SPLITS / "train.txt").read_text().split())
    eval_only = set()  # tiles in test_balanced - test
    if (SPLITS / "test_balanced.txt").exists():
        balanced = set((SPLITS / "test_balanced.txt").read_text().split())
        canonical_test = set((SPLITS / "test.txt").read_text().split())
        eval_only = balanced - canonical_test
    full = set((SPLITS / "train_full.txt").read_text().split()) if (SPLITS / "train_full.txt").exists() else set()
    extras = full - canonical_train - eval_only
    print(f"[curate] canonical train={len(canonical_train)}  extras={len(extras)}  eval_only={len(eval_only)}",
          flush=True)

    # If a cached scores file exists with all extras present, skip the expensive scoring pass.
    if SCORES_JSON.exists():
        cached = json.loads(SCORES_JSON.read_text())
        if set(cached.keys()) >= extras:
            scored = [(tid, cached[tid]["agr"], cached[tid]["ndvi"]) for tid in sorted(extras)]
            print(f"[curate] loaded cached scores for {len(scored)} extras from {SCORES_JSON.name}",
                  flush=True)
            return _bucket_and_write(scored, canonical_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    nc = int(sd.get("num_classes", 6))
    model = MultiHeadV3(num_classes=nc, encoder_name=sd.get("encoder_name", "mit_b3")).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()
    print(f"[curate] loaded {CKPT.name} (num_classes={nc})", flush=True)

    @torch.no_grad()
    def score_tile(tid: str) -> tuple[float, float, int, int]:
        """Return (3-class agreement, ndvi_mean, n_classes_in_mask, mask_size)."""
        ms, mask = load_tile(tid)
        # NDVI from raw bands
        red = ms[2] / 255.0  # restore 0..1 for ndvi calc
        nir = ms[4] / 255.0
        ndvi_mean = float(((nir - red) / (nir + red + 1e-6)).mean())
        # Resize ms+mask to 600 for model
        ms600 = to_native_600(ms)
        mask600 = to_native_600(mask.astype(np.float32)) if mask.shape[-2:] != (600, 600) else mask
        # Run model
        x = torch.from_numpy(ms600).float().unsqueeze(0).to(device)
        x_in = F.interpolate(x, size=(576, 576), mode="bilinear", align_corners=False) / 256.0
        x_in = x_in.clamp(0, 1)
        out = tta4_predict(model, x_in)
        seg = F.interpolate(out["seg"], size=(600, 600), mode="bilinear", align_corners=False)
        pred6 = seg.argmax(1).squeeze(0).cpu().numpy()
        pred3 = pred6.copy(); pred3[pred3 > 1] = 2
        m3 = mask600.copy() if isinstance(mask600, np.ndarray) else mask600
        if not isinstance(m3, np.ndarray): m3 = m3.numpy() if hasattr(m3, "numpy") else np.array(m3)
        m3 = m3.astype(np.int64); m3[m3 > 1] = 2
        agreement = float((pred3 == m3).mean())
        return agreement, ndvi_mean, len(np.unique(m3)), m3.size

    # Score each extra
    scored: list[tuple[str, float, float]] = []   # (tid, agreement, ndvi_mean)
    for i, tid in enumerate(sorted(extras)):
        try:
            agr, ndvi, _, _ = score_tile(tid)
            scored.append((tid, agr, ndvi))
            if (i + 1) % 50 == 0:
                print(f"[curate] {i+1}/{len(extras)}  last_tid={tid}  agr={agr:.3f}  ndvi={ndvi:+.3f}",
                      flush=True)
        except Exception as e:
            print(f"[curate] skip {tid}: {e}")

    # Cache scores so we can re-run the bucketing/picking step without re-running inference.
    SCORES_JSON.write_text(json.dumps(
        {tid: {"agr": agr, "ndvi": ndvi} for tid, agr, ndvi in scored}, indent=2))
    print(f"[curate] cached {len(scored)} scores -> {SCORES_JSON.name}", flush=True)

    return _bucket_and_write(scored, canonical_train)


def _bucket_and_write(scored: list[tuple[str, float, float]], canonical_train: set[str]):
    """Bucket scored tiles by NDVI, take top-K per bucket by agreement, write split."""
    buckets: dict[str, list[tuple[str, float, float]]] = {"healthy": [], "moderate": [], "low": []}
    for tid, agr, ndvi in scored:
        if ndvi >= 0.6: buckets["healthy"].append((tid, agr, ndvi))
        elif ndvi >= 0.4: buckets["moderate"].append((tid, agr, ndvi))
        else: buckets["low"].append((tid, agr, ndvi))

    chosen: list[str] = []
    for name, items in buckets.items():
        items.sort(key=lambda x: -x[1])
        target = PER_BUCKET_TARGETS[name]
        picked = items[:target]
        agr_top = [round(t[1], 2) for t in picked[:3]]
        agr_floor = round(picked[-1][1], 2) if picked else None
        print(f"[curate] {name:<9} bucket: {len(items)} total, top {len(picked)} kept "
              f"(top agr: {agr_top}, floor agr: {agr_floor})", flush=True)
        chosen.extend(t[0] for t in picked)

    train_curated = sorted(canonical_train | set(chosen))
    out = SPLITS / "train_curated.txt"
    out.write_text("\n".join(train_curated) + "\n")
    print(f"[curate] wrote {out.name}: {len(train_curated)} tiles "
          f"({len(canonical_train)} canonical + {len(chosen)} curated extras)", flush=True)


if __name__ == "__main__":
    main()
