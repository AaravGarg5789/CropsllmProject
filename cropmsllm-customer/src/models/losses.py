"""Loss functions for the v2 multi-task head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss. Ignores `ignore_index` if set."""

    def __init__(self, num_classes: int, eps: float = 1e-6, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        target_safe = target.clone()
        target_safe[~valid] = 0
        oh = F.one_hot(target_safe.long(), self.num_classes).permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1).float()
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + oh.sum(dim=dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class FocalCE(nn.Module):
    """Focal cross-entropy. Alpha=class weights (None=uniform), gamma=focal exponent."""

    def __init__(self, gamma: float = 2.0, weight: Tensor | None = None, ignore_index: int = -1) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        weight = self.weight if self.weight.numel() > 0 else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LovaszSoftmax(nn.Module):
    """Multi-class Lovász-Softmax loss (Berman et al., CVPR 2018) — direct
    surrogate for the IoU metric. Reliably gains 1-3 pp mIoU on small RS seg
    when stacked on dice + CE.
    """

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted.float()).cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        valid = (target != self.ignore_index)
        losses = []
        for c in range(C):
            fg = ((target == c) & valid).float().reshape(-1)
            if fg.sum() == 0:
                continue
            class_pred = probs[:, c].reshape(-1)[valid.reshape(-1)]
            class_fg = fg[valid.reshape(-1)]
            errors = (class_fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = class_fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        if not losses:
            return logits.new_zeros(())
        return torch.stack(losses).mean()


class KendallUncertaintyLoss(nn.Module):
    """Multi-task loss with learnable log-variance weighting (Kendall & Gal, CVPR 2018).

    L_total = sum_i (1 / 2σ_i²) · L_i + log σ_i, with log_sigma_i learnable.
    Avoids hand-tuning per-task scalars; the model learns task difficulty.
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[Tensor]) -> Tensor:
        assert len(losses) == self.log_sigma.numel()
        total = 0.0
        for i, l in enumerate(losses):
            precision = torch.exp(-2 * self.log_sigma[i])
            total = total + precision * l + self.log_sigma[i]
        return total
