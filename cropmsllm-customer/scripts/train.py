"""CropMS-LLM ViT - train.

Single canonical training entry point for the released ViT pipeline:

    Encoder       : MiT-B3 (SegFormer's hierarchical ViT), ImageNet init
    Decoder       : DeepLabv3+ (segmentation_models_pytorch)
    Heads (4)     : seg (6-class), dense NDVI, dense NDRE, scalar canopy-cover
    Loss          : Kendall-Gal uncertainty over (dice + class-wt focal-CE +
                    Lovász-Softmax) seg loss + L1 NDVI/NDRE/canopy aux losses
    Aug           : flip + rot90 + weed-instance copy-paste + MS-aware mixup
    Optimizer     : AdamW, weight decay 1e-4
    Schedule      : 5 warmup epochs (encoder frozen, lr 1e-3) +
                    35 main epochs (full unfreeze, lr 3e-4 cosine to 1e-6 with
                    5% linear warmup; per-phase counter, no wrap)
    EMA           : decay 0.99 with 50-step verbatim warmup
    Train res     : 576x576 (MiT requires multiples of 32; eval upscales to 600)
    VRAM          : peak ~6.4 GB on RTX 4060 8 GB at batch=4

Saves: runs/cropmsllm_vit.pt (best val 6-class mIoU).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.augment_v3 import ms_aware_mixup, weed_instance_copy_paste
from src.data.weedsgalore import WeedsGalore
from src.models.losses import DiceLoss, FocalCE, KendallUncertaintyLoss, LovaszSoftmax
from src.models.multi_head import compute_supervision_targets
from src.models.multi_head_v3 import MultiHeadV3


CLASS_NAMES_6 = ["background", "maize", "weed_a", "weed_b", "weed_c", "weed_d"]
CLASS_NAMES_3 = ["background", "maize", "weed"]
DATA_ROOT = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"   # data/ at project root
ENCODER_NAME = "mit_b3"
TRAIN_HW = (576, 576)


NATIVE_HW = (600, 600)   # canonical post-augmentation tile size - orthomosaic _om_
                          # tiles arrive at 224×224; bilinearly upsample so collate stacks


def _ensure_native(ms, mask):
    """Resize a single (5,H,W) tile + (H,W) mask to NATIVE_HW if needed."""
    if not torch.is_tensor(ms): ms = torch.from_numpy(ms).float()
    if not torch.is_tensor(mask): mask = torch.from_numpy(mask).long()
    if (ms.shape[-2], ms.shape[-1]) != NATIVE_HW:
        ms = F.interpolate(ms.unsqueeze(0), size=NATIVE_HW, mode="bilinear",
                           align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                             size=NATIVE_HW, mode="nearest").squeeze().long()
    return ms, mask


def _maybe_merge_3class(mask: torch.Tensor) -> torch.Tensor:
    """Map weed_a/b/c/d (2,3,4,5) → 2 for the WACV uni-weed metric. Idempotent."""
    out = mask.clone()
    out[out > 1] = 2
    return out


def aug_train(ms, mask):
    import random
    ms, mask = _ensure_native(ms, mask)
    if random.random() < 0.5:
        ms = torch.flip(ms, dims=(-1,)); mask = torch.flip(mask, dims=(-1,))
    if random.random() < 0.5:
        ms = torch.flip(ms, dims=(-2,)); mask = torch.flip(mask, dims=(-2,))
    k = random.randint(0, 3)
    if k:
        ms = torch.rot90(ms, k=k, dims=(-2, -1))
        mask = torch.rot90(mask, k=k, dims=(-2, -1))
    return ms, mask


def aug_val(ms, mask):
    return _ensure_native(ms, mask)


def to_train_size(ms, mask=None):
    if (ms.shape[-2], ms.shape[-1]) != TRAIN_HW:
        ms = F.interpolate(ms, size=TRAIN_HW, mode="bilinear", align_corners=False)
    if mask is not None and mask.shape[-2:] != TRAIN_HW:
        mask = F.interpolate(mask.unsqueeze(1).float(), size=TRAIN_HW, mode="nearest").squeeze(1).long()
    return ms, mask


def normalize_ms(ms): return (ms / 256.0).clamp(0, 1)


def class_weights_from_train(num_classes, mode, merge_3=False):
    if mode == "none":
        return torch.ones(num_classes)
    counts = np.zeros(num_classes, dtype=np.int64)
    ds = WeedsGalore(DATA_ROOT, "train")
    for i in range(len(ds)):
        m = ds[i].mask.numpy() if torch.is_tensor(ds[i].mask) else ds[i].mask
        if merge_3:
            m = m.copy(); m[m > 1] = 2
        c = np.bincount(m.ravel(), minlength=num_classes)
        counts += c[:num_classes]
    freq = counts / counts.sum()
    if mode == "inv-freq":
        w = 1.0 / np.maximum(freq, 1e-6)
    elif mode == "sqrt-inv-freq":
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    else:
        raise ValueError(mode)
    w = w / w.mean()
    return torch.from_numpy(np.clip(w, 0.0, 10.0)).float()


@torch.no_grad()
def evaluate(model, loader, device, num_classes=6, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES_6 if num_classes == 6 else CLASS_NAMES_3
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    correct = 0; total = 0
    for ms, mask in loader:
        ms = ms.to(device); mask = mask.to(device)
        ms_t, _ = to_train_size(ms, None)
        x = normalize_ms(ms_t)
        out = model(x)
        H, W = mask.shape[-2:]
        seg_up = F.interpolate(out["seg"], size=(H, W), mode="bilinear", align_corners=False)
        pred = seg_up.argmax(1)
        correct += int((pred == mask).sum()); total += int(mask.numel())
        for c in range(num_classes):
            inter[c] += ((pred == c) & (mask == c)).sum()
            union[c] += ((pred == c) | (mask == c)).sum()
    iou = (inter / union.clamp(min=1)).cpu().tolist()
    label = "3class" if num_classes == 3 else "6class"
    return {
        f"mean_iou_{label}": sum(iou) / len(iou),
        "pixel_accuracy": correct / max(total, 1),
        "per_class_iou": dict(zip(class_names, iou)),
    }


def warmup_cosine_lr(step, total_steps, warmup_frac, base_lr, min_lr):
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-warmup", type=int, default=5)
    ap.add_argument("--epochs-main", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-warmup", type=float, default=1e-3)
    ap.add_argument("--lr-main", type=float, default=3e-4)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--ema-warmup-steps", type=int, default=50)
    ap.add_argument("--class-weights", type=str, default="sqrt-inv-freq",
                    choices=["none", "inv-freq", "sqrt-inv-freq"])
    ap.add_argument("--cp-prob", type=float, default=0.5)
    ap.add_argument("--mixup-prob", type=float, default=0.3)
    ap.add_argument("--mixup-lam-max", type=float, default=0.3)
    ap.add_argument("--lovasz-weight", type=float, default=0.5)
    ap.add_argument("--train-split", type=str, default="train",
                    help="train / train_extended - pick which split file to read")
    ap.add_argument("--ckpt-name", type=str, default="cropmsllm_vit",
                    help="checkpoint base name; saves runs/<name>.pt")
    ap.add_argument("--num-classes", type=int, default=6, choices=[3, 6],
                    help="3 = bg/maize/merged-weed (WACV uni-weed metric); 6 = full multi-weed")
    args = ap.parse_args()
    num_classes = args.num_classes
    class_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_6
    merge_to_3class = num_classes == 3
    print(f"[train] num_classes={num_classes}  classes={class_names}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    cw = class_weights_from_train(num_classes, args.class_weights, merge_3=merge_to_3class).to(device)
    print(f"[class_weights] {args.class_weights} -> {[f'{w:.2f}' for w in cw.cpu().tolist()]}", flush=True)

    # Wrap aug functions so they optionally merge to 3-class targets
    if merge_to_3class:
        def aug_train_wrapped(ms, mask):
            ms, mask = aug_train(ms, mask)
            return ms, _maybe_merge_3class(mask)
        def aug_val_wrapped(ms, mask):
            ms, mask = aug_val(ms, mask)
            return ms, _maybe_merge_3class(mask)
    else:
        aug_train_wrapped, aug_val_wrapped = aug_train, aug_val

    train_ds = WeedsGalore(DATA_ROOT, args.train_split, transform=aug_train_wrapped)
    val_ds = WeedsGalore(DATA_ROOT, "val", transform=aug_val_wrapped)
    bank_ds = WeedsGalore(DATA_ROOT, args.train_split, transform=aug_val_wrapped)
    out_ckpt = REPO / "runs" / f"{args.ckpt_name}.pt"
    log_path = REPO / "runs" / f"train_{args.ckpt_name}.log.json"
    print(f"[train] split={args.train_split}  n_train={len(train_ds)}  ckpt -> {out_ckpt.name}",
          flush=True)

    def collate_train(samples):
        import random
        out_ms, out_masks = [], []
        for s in samples:
            ms, m = s.ms, s.mask
            if random.random() < args.cp_prob:
                src = bank_ds[random.randrange(len(bank_ds))]
                ms, m = weed_instance_copy_paste(ms, m, src.ms, src.mask, p=1.0)
            if random.random() < args.mixup_prob and args.mixup_lam_max > 0:
                src = bank_ds[random.randrange(len(bank_ds))]
                ms, m = ms_aware_mixup(ms, m, src.ms, src.mask, num_classes=num_classes,
                                        p=1.0, lam_max=args.mixup_lam_max)
            out_ms.append(ms); out_masks.append(m)
        return torch.stack(out_ms), torch.stack(out_masks)

    # drop_last avoids ASPP-BN crash on a trailing batch of 1 sample (BN needs >=2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_train, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda s: (torch.stack([t.ms for t in s]),
                                                  torch.stack([t.mask for t in s])),
                            num_workers=0)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}  encoder={ENCODER_NAME} "
          f"train_hw={TRAIN_HW}", flush=True)

    model = MultiHeadV3(num_classes=num_classes, encoder_name=ENCODER_NAME).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)

    seg_dice = DiceLoss(num_classes).to(device)
    seg_focal = FocalCE(gamma=2.0, weight=cw).to(device)
    seg_lovasz = LovaszSoftmax().to(device)
    kendall = KendallUncertaintyLoss(n_tasks=4).to(device)

    history = []
    best_miou = -1.0
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    global_step = 0

    def run_phase(phase, epochs, lr):
        nonlocal best_miou, global_step
        model.freeze_encoder(phase == "warmup")
        params = [p for p in model.parameters() if p.requires_grad] + list(kendall.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        total_steps = epochs * len(train_loader)
        warmup_frac = 0.05 if phase == "main" else 0.0
        phase_step = 0

        for ep in range(epochs):
            model.train()
            t0 = time.time()
            agg = {"loss": 0.0, "seg": 0.0, "ndvi": 0.0, "ndre": 0.0, "cc": 0.0}
            for ms, mask in train_loader:
                ms = ms.to(device); mask = mask.to(device)
                ms, mask = to_train_size(ms, mask)
                x = normalize_ms(ms)
                tgt = compute_supervision_targets(x, mask)

                cur_lr = warmup_cosine_lr(phase_step, total_steps, warmup_frac, lr, lr / 100.0)
                for g in opt.param_groups: g["lr"] = cur_lr

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    out = model(x)
                    L_seg = (seg_dice(out["seg"], mask)
                             + seg_focal(out["seg"], mask)
                             + args.lovasz_weight * seg_lovasz(out["seg"], mask))
                    L_ndvi = (out["ndvi"] - tgt["ndvi"]).abs().mean()
                    L_ndre = (out["ndre"] - tgt["ndre"]).abs().mean()
                    L_cc = (out["canopy"] - tgt["canopy"]).abs().mean()
                    L = kendall([L_seg, L_ndvi, L_ndre, L_cc])
                if scaler is not None:
                    scaler.scale(L).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(opt); scaler.update()
                else:
                    L.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt.step()

                global_step += 1; phase_step += 1
                with torch.no_grad():
                    if global_step <= args.ema_warmup_steps:
                        for ep_, p_ in zip(ema.parameters(), model.parameters()):
                            ep_.copy_(p_.detach())
                        for eb_, mb_ in zip(ema.buffers(), model.buffers()):
                            eb_.copy_(mb_.detach())
                    else:
                        for ep_, p_ in zip(ema.parameters(), model.parameters()):
                            ep_.mul_(args.ema_decay).add_(p_.detach(), alpha=1 - args.ema_decay)
                        for eb_, mb_ in zip(ema.buffers(), model.buffers()):
                            eb_.copy_(mb_.detach())

                agg["loss"] += float(L); agg["seg"] += float(L_seg)
                agg["ndvi"] += float(L_ndvi); agg["ndre"] += float(L_ndre); agg["cc"] += float(L_cc)
            for k in agg: agg[k] /= max(len(train_loader), 1)

            metrics = evaluate(ema, val_loader, device, num_classes=num_classes,
                                class_names=class_names)
            label = "3-cls" if num_classes == 3 else "6-cls"
            miou_key = f"mean_iou_{'3class' if num_classes == 3 else '6class'}"
            dt = time.time() - t0
            print(f"[{phase} ep {ep+1:02d}/{epochs}] {dt:.1f}s lr={cur_lr:.2e}  L={agg['loss']:.3f} "
                  f"seg={agg['seg']:.3f} | val {label} mIoU={metrics[miou_key]*100:.2f}% "
                  f"pix={metrics['pixel_accuracy']*100:.2f}%", flush=True)
            history.append({"phase": phase, "epoch": ep + 1, "train": agg, "val": metrics})
            if metrics[miou_key] > best_miou:
                best_miou = metrics[miou_key]
                out_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": ema.state_dict(),
                    "encoder_name": ENCODER_NAME,
                    "kendall_state_dict": kendall.state_dict(),
                    "num_classes": num_classes,
                    f"best_val_{miou_key}": best_miou,
                    "best_val_metrics": {k: v for k, v in metrics.items() if k != "per_class_iou"},
                    "epoch": ep + 1, "phase": phase,
                    "train_hw": list(TRAIN_HW),
                }, out_ckpt)
                print(f"  -> saved {out_ckpt.name} (best val {label} mIoU = {best_miou*100:.2f}%)",
                      flush=True)

    print(f"[stage A] {args.epochs_warmup} epochs frozen, lr={args.lr_warmup}", flush=True)
    run_phase("warmup", args.epochs_warmup, args.lr_warmup)
    if device == "cuda": torch.cuda.empty_cache()
    print(f"[stage B] {args.epochs_main} epochs unfrozen, lr={args.lr_main}", flush=True)
    run_phase("main", args.epochs_main, args.lr_main)

    log_path.write_text(json.dumps(history, indent=2))
    if device == "cuda":
        print(f"[vram] peak alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    label = "3-cls" if num_classes == 3 else "6-cls"
    print(f"[done] best val {label} mIoU = {best_miou*100:.2f}%; log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
