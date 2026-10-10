"""CropMS-LLM ViT - eval.

Evaluates the trained MiT-B3 multi-head model on train/val/test under both:
  * 3-class mIoU (uni-weed: bg / crop / merged-weed) - WACV 2025 headline
  * 6-class mIoU (multi-weed: bg / maize / weed_a / weed_b / weed_c / weed_d)
plus pixel accuracy, NDVI MAE, NDRE MAE, canopy MAE.

Uses 4-flip TTA at eval time. Writes runs/eval_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.weedsgalore import WeedsGalore
from src.inference.tta import tta4_predict
from src.models.multi_head import compute_supervision_targets
from src.models.multi_head_v3 import MultiHeadV3


CLASS_NAMES = ["background", "maize", "weed_a", "weed_b", "weed_c", "weed_d"]
NUM_CLASSES = 6
DATA_ROOT = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"
CKPT_DEFAULT = REPO / "runs" / "cropmsllm_vit.pt"
OUT_DEFAULT = REPO / "runs" / "eval_metrics.json"
WACV_BASELINE = 0.829


def to_t(ms, mask):
    if not torch.is_tensor(ms): ms = torch.from_numpy(ms).float()
    if not torch.is_tensor(mask): mask = torch.from_numpy(mask).long()
    return ms, mask


def _ensure_native(ms, mask, target_hw=(600, 600)):
    """Resize 224×224 _om_ tiles up to native 600×600 (matches trainer convention)."""
    if (ms.shape[-2], ms.shape[-1]) != target_hw:
        ms = F.interpolate(ms.unsqueeze(0), size=target_hw, mode="bilinear",
                            align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_hw,
                              mode="nearest").squeeze().long()
    return ms, mask


# --- Per-task derivation rules (must match release/src/inference/task_outputs.py) ---
NDVI_HEALTHY = 0.6
NDRE_NDEF    = 0.20
NDVI_VEG     = 0.30
NDVI_WATER   = 0.30
NDVI_GOOD    = 0.55


def _stress_type_map(ndvi, ndre, nir):
    """(B, H, W) for each input → (B, H, W) integer map in {0..3}."""
    nir_p50 = nir.flatten(1).quantile(0.5, dim=1).view(-1, 1, 1)
    n_def   = (ndre < NDRE_NDEF) & (ndvi > NDVI_VEG)
    water   = (ndvi < NDVI_WATER) & (nir < nir_p50)
    general = (ndvi < 0.4) & ~n_def & ~water
    out = torch.zeros_like(ndvi, dtype=torch.long)
    out[n_def] = 1; out[water] = 2; out[general] = 3
    return out


def _zone_map(ndvi, seg):
    """(B, H, W) → 0=bare, 1=poor, 2=good."""
    out = torch.ones_like(ndvi, dtype=torch.long)
    out[ndvi >= NDVI_GOOD] = 2
    out[seg == 0] = 0
    return out


def _binary_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-9) if (tp + fp + fn) > 0 else 0.0)
    return sum(f1s) / max(len(f1s), 1)


@torch.no_grad()
def eval_split(model, split, train_hw, device, model_classes=6):
    """`model_classes` is the seg-head output dimension (3 = WACV uni-weed, 6 = full).
    For a 3-class model the per-class-IoU on a 6-class scale isn't computable; only
    the 3-class metric is reported."""
    ds = WeedsGalore(DATA_ROOT, split, transform=lambda ms, mask: _ensure_native(*to_t(ms, mask)))
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                       collate_fn=lambda s: (torch.stack([t.ms for t in s]),
                                             torch.stack([t.mask for t in s])))
    inter6 = torch.zeros(NUM_CLASSES, device=device); union6 = torch.zeros(NUM_CLASSES, device=device)
    inter3 = torch.zeros(3, device=device); union3 = torch.zeros(3, device=device)
    inter_zone = torch.zeros(3, device=device); union_zone = torch.zeros(3, device=device)
    correct = 0; total = 0
    ndvi_mae = 0.0; ndre_mae = 0.0; cc_mae = 0.0; n = 0
    ndvi_sse = 0.0; ndvi_count = 0
    health_true: list[int] = []; health_pred: list[int] = []
    stress_correct = 0; stress_total = 0
    stress_true_all: list[torch.Tensor] = []; stress_pred_all: list[torch.Tensor] = []

    for ms, mask in loader:
        ms = ms.to(device); mask = mask.to(device)
        ms_t = F.interpolate(ms, size=train_hw, mode="bilinear", align_corners=False)
        x = (ms_t / 256.0).clamp(0, 1)
        out = tta4_predict(model, x)
        H, W = mask.shape[-2:]
        seg_up = F.interpolate(out["seg"], size=(H, W), mode="bilinear", align_corners=False)
        ndvi_up = F.interpolate(out["ndvi"], size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        ndre_up = F.interpolate(out["ndre"], size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        # ---- Task: 6-class & 3-class segmentation IoU ----
        pred = seg_up.argmax(1)
        if model_classes == 3:
            # Model outputs 3 classes natively → already merged-weed
            pred3 = pred
            pred6 = None
        else:
            pred6 = pred
            pred3 = pred.clone(); pred3[pred3 > 1] = 2
        mask3 = mask.clone(); mask3[mask3 > 1] = 2
        # IoU on the model's native dimensionality
        correct += int((pred3 == mask3).sum()); total += int(mask.numel())
        for c in range(3):
            inter3[c] += ((pred3 == c) & (mask3 == c)).sum()
            union3[c] += ((pred3 == c) | (mask3 == c)).sum()
        if pred6 is not None:
            for c in range(NUM_CLASSES):
                inter6[c] += ((pred6 == c) & (mask == c)).sum()
                union6[c] += ((pred6 == c) | (mask == c)).sum()

        # ---- Task: NDVI / NDRE / canopy regression (MAE + RMSE) ----
        x_native = (ms / 256.0).clamp(0, 1)
        tgt = compute_supervision_targets(x_native, mask)
        ndvi_true = tgt["ndvi"].squeeze(1)
        ndre_true = tgt["ndre"].squeeze(1)
        ndvi_mae += float((ndvi_up - ndvi_true).abs().mean()) * ms.size(0)
        ndre_mae += float((ndre_up - ndre_true).abs().mean()) * ms.size(0)
        cc_mae += float((out["canopy"] - tgt["canopy"]).abs().mean()) * ms.size(0)
        n += ms.size(0)
        ndvi_sse += float(((ndvi_up - ndvi_true) ** 2).sum())
        ndvi_count += ndvi_up.numel()

        # ---- Task 1: Crop Health (binary, per tile) ----
        # "True health"  = mean(true NDVI over true vegetation pixels) ≥ 0.6
        # "Pred health"  = mean(pred NDVI over pred vegetation pixels) ≥ 0.6
        for b in range(ms.size(0)):
            true_veg_mask = (mask[b] > 0)
            pred_veg_mask = (pred[b] > 0)            # works for both 3-cls and 6-cls
            if true_veg_mask.any():
                t_score = float(ndvi_true[b][true_veg_mask].mean())
            else:
                t_score = float(ndvi_true[b].mean())
            if pred_veg_mask.any():
                p_score = float(ndvi_up[b][pred_veg_mask].mean())
            else:
                p_score = float(ndvi_up[b].mean())
            health_true.append(int(t_score >= NDVI_HEALTHY))
            health_pred.append(int(p_score >= NDVI_HEALTHY))

        # ---- Task 3: Stress type (per-pixel 4-class: 0=none,1=Ndef,2=Water,3=Gen) ----
        nir_native = ms[:, 4]
        stress_true = _stress_type_map(ndvi_true, ndre_true, nir_native)
        stress_pred = _stress_type_map(ndvi_up,   ndre_up,   nir_native)
        stress_correct += int((stress_true == stress_pred).sum())
        stress_total += int(stress_true.numel())
        stress_true_all.append(stress_true.flatten().cpu())
        stress_pred_all.append(stress_pred.flatten().cpu())

        # ---- Task 4: Field zoning (per-pixel 3-class: 0=bare,1=poor,2=good) ----
        zone_true = _zone_map(ndvi_true, mask)
        zone_pred = _zone_map(ndvi_up, pred)         # uses model's native pred (3 or 6)
        for c in range(3):
            inter_zone[c] += ((zone_pred == c) & (zone_true == c)).sum()
            union_zone[c] += ((zone_pred == c) | (zone_true == c)).sum()

    # ---- collate ----
    iou6 = (inter6 / union6.clamp(min=1)).cpu().tolist()
    iou3 = (inter3 / union3.clamp(min=1)).cpu().tolist()
    iou_zone = (inter_zone / union_zone.clamp(min=1)).cpu().tolist()

    # subsample stress vectors to limit memory before sklearn-style F1
    st_true = torch.cat(stress_true_all)
    st_pred = torch.cat(stress_pred_all)
    if st_true.numel() > 4_000_000:
        idx = torch.randperm(st_true.numel())[:4_000_000]
        st_true, st_pred = st_true[idx], st_pred[idx]

    return {
        "n_tiles": len(ds),
        "pixel_accuracy": correct / max(total, 1),
        "model_classes": model_classes,
        "mean_iou_3class": sum(iou3) / len(iou3),
        "mean_iou_6class": (sum(iou6) / len(iou6)) if model_classes == 6 else None,
        "iou_3class": {"bg": iou3[0], "crop": iou3[1], "weed": iou3[2]},
        "per_class_iou_6class": dict(zip(CLASS_NAMES, iou6)) if model_classes == 6 else None,
        "ndvi_mae": ndvi_mae / max(n, 1),
        "ndre_mae": ndre_mae / max(n, 1),
        "canopy_mae": cc_mae / max(n, 1),
        # Per-task metrics matching the multi-task spec
        "task_health_f1":         _binary_f1(torch.tensor(health_true), torch.tensor(health_pred)),
        "task_stress_accuracy":   stress_correct / max(stress_total, 1),
        "task_stress_macro_f1":   _macro_f1(st_true, st_pred, num_classes=4),
        "task_ndvi_rmse":         (ndvi_sse / max(ndvi_count, 1)) ** 0.5,
        "task_zone_mean_iou":     sum(iou_zone) / len(iou_zone),
        "task_zone_per_class_iou": {"bare": iou_zone[0], "poor": iou_zone[1], "good": iou_zone[2]},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--splits", nargs="+", default=("train", "val", "test"),
                    help="which splits to eval (e.g. train val test test_balanced)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"[skip] {args.ckpt.name} not found - train first via scripts/train.py")
        return 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder = sd.get("encoder_name", "mit_b3")
    train_hw = tuple(sd.get("train_hw", (576, 576)))
    nc = int(sd.get("num_classes", NUM_CLASSES))
    model = MultiHeadV3(num_classes=nc, encoder_name=encoder).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()
    print(f"[eval] ckpt={args.ckpt.name}  encoder={encoder}  num_classes={nc}  train_hw={train_hw}")

    out = {"checkpoint": str(args.ckpt), "device": device, "encoder": encoder,
           "num_classes": nc, "train_hw": list(train_hw), "tta": "4x flips",
           "splits": {}}
    for split in args.splits:
        print(f"[eval] {split} ...")
        m = eval_split(model, split, train_hw, device, model_classes=nc)
        out["splits"][split] = m
        m6 = m["mean_iou_6class"]
        m6_str = f"{m6*100:.2f}%" if m6 is not None else "-"
        print(f"  n={m['n_tiles']}  3-cls mIoU={m['mean_iou_3class']*100:.2f}%  "
              f"6-cls mIoU={m6_str}  "
              f"pix_acc={m['pixel_accuracy']*100:.2f}%  "
              f"ndvi_mae={m['ndvi_mae']:.3f}  ndre_mae={m['ndre_mae']:.3f}  "
              f"health-F1 {m['task_health_f1']*100:.2f}%  zone-mIoU {m['task_zone_mean_iou']*100:.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
