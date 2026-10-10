"""v3-only augmentation utilities: weed-instance copy-paste + MS-aware mixup.

These augment a single training sample using a *bank* of (ms, mask) tensors
sampled from elsewhere in the train split. The trainer is responsible for
maintaining the bank and calling these helpers; we keep them here as pure
functions so they're easy to unit-test.

Both keep the 5-band MS tensor and integer mask aligned (whatever
spatial transform happens to the MS, the same happens to the mask).

The MS-aware mixup helper does NOT require the caller to recompute NDVI/NDRE
targets — those are derived from the (now blended) MS bands at supervision
time via `compute_supervision_targets` in `src/models/multi_head.py`. So the
invariant is automatically preserved.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from scipy import ndimage as ndi
from torch import Tensor


WEED_CLASSES: tuple[int, ...] = (2, 3, 4, 5)        # weed_a, weed_b, weed_c, weed_d
BG_CLASS: int = 0


def _to_tensors(ms, mask) -> tuple[Tensor, Tensor]:
    if not torch.is_tensor(ms):
        ms = torch.from_numpy(np.asarray(ms)).float()
    if not torch.is_tensor(mask):
        mask = torch.from_numpy(np.asarray(mask)).long()
    return ms, mask


def weed_instance_copy_paste(
    target_ms: Tensor, target_mask: Tensor,
    source_ms: Tensor, source_mask: Tensor,
    p: float = 0.5,
    min_pixels: int = 80,
    max_paste_per_call: int = 2,
    class_weights: dict[int, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Cut a connected weed component from `source` and paste at a valid spot
    on `target`. "Valid spot" = a region of the target currently labeled as
    background (we don't paste over existing crop / weed pixels).

    Returns the augmented (ms, mask). If no eligible component is found or
    `random.random() >= p`, returns target unchanged.
    """
    if random.random() >= p:
        return target_ms, target_mask

    target_ms, target_mask = _to_tensors(target_ms, target_mask)
    source_ms, source_mask = _to_tensors(source_ms, source_mask)

    # Find weed components in the source.
    src_np = source_mask.cpu().numpy()
    weed_mask_np = np.isin(src_np, WEED_CLASSES)
    if not weed_mask_np.any():
        return target_ms, target_mask

    labels, n = ndi.label(weed_mask_np)
    if n == 0:
        return target_ms, target_mask

    # Pick valid components (>= min_pixels). If class_weights provided, prefer rare classes.
    sizes = ndi.sum_labels(weed_mask_np.astype(np.int32), labels, index=range(1, n + 1))
    valid_ids = [i + 1 for i, sz in enumerate(sizes) if sz >= min_pixels]
    if not valid_ids:
        return target_ms, target_mask
    if class_weights is None:
        random.shuffle(valid_ids)
    else:
        # weighted sample without replacement (top-K by class weight, then shuffle)
        weights = []
        for cid in valid_ids:
            ys, xs = np.where(labels == cid)
            comp_class = int(src_np[ys[0], xs[0]])           # all pixels share class
            weights.append(class_weights.get(comp_class, 1.0))
        order = sorted(range(len(valid_ids)), key=lambda i: -weights[i] * random.random())
        valid_ids = [valid_ids[i] for i in order]

    pasted = 0
    H, W = target_mask.shape[-2], target_mask.shape[-1]
    bg = (target_mask == BG_CLASS).cpu().numpy()

    for cid in valid_ids:
        if pasted >= max_paste_per_call:
            break
        ys, xs = np.where(labels == cid)
        if len(ys) == 0: continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0
        if h >= H or w >= W:
            continue
        # try up to 8 random positions to find background-only landing pad
        for _ in range(8):
            ty = random.randint(0, H - h)
            tx = random.randint(0, W - w)
            if bg[ty:ty + h, tx:tx + w].mean() < 0.85:
                continue
            comp_mask = (labels[y0:y1, x0:x1] == cid)        # bool (h, w)
            comp_t = torch.from_numpy(comp_mask)
            target_ms[..., ty:ty + h, tx:tx + w] = torch.where(
                comp_t.unsqueeze(0),
                source_ms[..., y0:y1, x0:x1],
                target_ms[..., ty:ty + h, tx:tx + w],
            )
            target_mask[..., ty:ty + h, tx:tx + w] = torch.where(
                comp_t,
                source_mask[..., y0:y1, x0:x1],
                target_mask[..., ty:ty + h, tx:tx + w],
            )
            bg[ty:ty + h, tx:tx + w] &= ~comp_mask
            pasted += 1
            break

    return target_ms, target_mask


def ms_aware_mixup(
    a_ms: Tensor, a_mask: Tensor,
    b_ms: Tensor, b_mask: Tensor,
    num_classes: int = 6,
    p: float = 0.3,
    lam_max: float = 0.3,
) -> tuple[Tensor, Tensor]:
    """Blend two tiles' MS inputs and one-hot masks. Returns (blended_ms, soft_mask).

    The trainer should accept a soft (num_classes, H, W) target by switching
    to a soft-label CE on the seg head. To keep the existing hard-label trainer
    unchanged, this returns a HARD argmax mask of the soft blend instead — a
    less aggressive form of mixup that still smooths weed/background boundaries
    when lam_max stays small.

    NDVI/NDRE supervision is derived from the blended MS (still valid since
    NDVI is band-ratio of the blend, computed by the trainer).
    """
    if random.random() >= p or lam_max <= 0:
        return _to_tensors(a_ms, a_mask)
    a_ms, a_mask = _to_tensors(a_ms, a_mask)
    b_ms, b_mask = _to_tensors(b_ms, b_mask)
    lam = random.uniform(0.0, lam_max)
    blended_ms = (1 - lam) * a_ms + lam * b_ms
    a_oh = torch.nn.functional.one_hot(a_mask.long(), num_classes).permute(2, 0, 1).float()
    b_oh = torch.nn.functional.one_hot(b_mask.long(), num_classes).permute(2, 0, 1).float()
    soft = (1 - lam) * a_oh + lam * b_oh
    hard = soft.argmax(dim=0).long()
    return blended_ms, hard
