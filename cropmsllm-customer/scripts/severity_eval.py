"""Compute 3-class severity-bin classification metrics on canonical test split
(Severe NDVI<0.2 / Moderate 0.2-0.4 / Mild >=0.4) and persist to disk so the
metrics panel can render them without re-running inference."""
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.inference.pipeline_v3 import V3Predictor
from src.inference.pipeline import load_ms_tile

DATA = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"
CKPT = REPO / "runs" / "cropmsllm_vit.pt"
predictor = V3Predictor(CKPT, use_tta=True)

test_tiles = [ln.strip() for ln in open(DATA / "splits" / "test.txt") if ln.strip()]

BINS_4 = (0.2, 0.4, 0.6)
BINS_3 = (0.2, 0.4)

def bin_acc(pred_ndvi, actual_ndvi, bins):
    p = np.digitize(pred_ndvi, bins)
    a = np.digitize(actual_ndvi, bins)
    return float((p == a).mean()), p, a

agree_3, agree_4 = [], []
confusion_3 = np.zeros((3, 3), dtype=np.int64)
for tid in test_tiles:
    date = tid[:10]
    ms_path = DATA / date / "images" / f"{tid}_R.png"
    ms, _, _ = load_ms_tile(ms_path)
    red = ms[2].astype(np.float32) / 255.0
    nir = ms[4].astype(np.float32) / 255.0
    actual = (nir - red) / (nir + red + 1e-6)
    pred = predictor.predict(ms)["ndvi_pred"]
    a3, p3, t3 = bin_acc(pred, actual, BINS_3)
    a4, _, _   = bin_acc(pred, actual, BINS_4)
    agree_3.append(a3); agree_4.append(a4)
    for pp, tt in zip(p3.flatten(), t3.flatten()):
        confusion_3[tt, pp] += 1

print(f"3-class severity-bin per-pixel accuracy ({len(test_tiles)} tiles):")
print(f"  mean = {np.mean(agree_3)*100:.2f}%")
print(f"  per-tile [min/med/max] = "
      f"[{np.min(agree_3)*100:.1f}/{np.median(agree_3)*100:.1f}/{np.max(agree_3)*100:.1f}]%")
print(f"\n4-class severity-bin per-pixel accuracy: mean = {np.mean(agree_4)*100:.2f}%")

print("\n3-class confusion matrix (rows=actual, cols=predicted):")
print("                Severe   Moderate    Mild")
N = confusion_3.sum()
for i, name in enumerate(["Severe  ", "Moderate", "Mild    "]):
    row = confusion_3[i]
    rs = row.sum()
    pcts = [f"{r/max(rs,1)*100:6.1f}%" for r in row]
    print(f"  {name}  {pcts[0]}  {pcts[1]}  {pcts[2]}   ({rs/N*100:.1f}% of total)")

print("\n3-class per-bin precision / recall / F1:")
per_class: dict[str, dict[str, float]] = {}
for i, name in enumerate(["Severe", "Moderate", "Mild"]):
    tp = confusion_3[i, i]
    fp = confusion_3[:, i].sum() - tp
    fn = confusion_3[i, :].sum() - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    per_class[name] = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    print(f"  {name:<10}  P={p*100:6.2f}%   R={r*100:6.2f}%   F1={f1*100:6.2f}%")

macro_f1 = float(np.mean([per_class[n]["f1"] for n in ("Severe", "Moderate", "Mild")]))
print(f"\nMacro-F1 (3-class) = {macro_f1*100:.2f}%")

# Persist for the API metrics panel.
out = REPO / "runs" / "severity_metrics.json"
out.write_text(json.dumps({
    "split": "test",
    "n_tiles": int(len(test_tiles)),
    "bins_3class": {"severe": "<0.2", "moderate": "0.2-0.4", "mild": ">=0.4"},
    "per_pixel_accuracy": {
        "mean_3class": float(np.mean(agree_3)),
        "min_3class":  float(np.min(agree_3)),
        "median_3class": float(np.median(agree_3)),
        "max_3class": float(np.max(agree_3)),
        "mean_4class": float(np.mean(agree_4)),
    },
    "per_class": per_class,
    "macro_f1_3class": macro_f1,
    "pixel_distribution_pct": {
        "severe":   float(confusion_3[0].sum() / N * 100),
        "moderate": float(confusion_3[1].sum() / N * 100),
        "mild":     float(confusion_3[2].sum() / N * 100),
    },
}, indent=2))
print(f"\nWrote {out.relative_to(REPO)}")
