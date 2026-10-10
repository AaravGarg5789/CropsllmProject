"""MultiHeadV3 single-tile inference wrapper.

Honors the `train_hw` saved in the checkpoint (set by `scripts/train.py`):
hierarchical-ViT encoders (MiT, ConvNeXt) need a fixed multiple-of-32 input,
so we resize the input to `train_hw`, run the model, then bilinearly upscale
predictions back to the input's native HxW.

For dilated-CNN encoders (ResNet50/101/etc.), `train_hw` is `None` and we
forward at the native input resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.tta import tta4_predict
from src.models.multi_head_v3 import MultiHeadV3


NUM_CLASSES = 6
_VIT_LIKE_ENCODERS = ("mit_", "tu-mit_", "tu-convnext", "tu-swin", "tu-vit")


def _needs_resize(encoder: str) -> bool:
    return any(encoder.startswith(p) for p in _VIT_LIKE_ENCODERS)


class V3Predictor:
    def __init__(self, ckpt_path: Path, device: str | None = None,
                 use_tta: bool = True) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        encoder = sd.get("encoder_name", "resnet50")
        # Read num_classes from checkpoint metadata (3 = WACV uni-weed, 6 = full)
        # — older ckpts default to 6.
        nc = int(sd.get("num_classes", NUM_CLASSES))
        self.num_classes = nc
        self.model = MultiHeadV3(num_classes=nc, encoder_name=encoder).to(self.device)
        self.model.load_state_dict(sd["model_state_dict"])
        self.model.eval()
        # `best_val_miou` retained for backwards compat across ckpt-key formats:
        # - very old: best_val_miou
        # - 6-class: best_val_miou_6class / best_val_mean_iou_6class
        # - 3-class: best_val_mean_iou_3class
        self.best_val_miou = (sd.get("best_val_mean_iou_3class")
                              or sd.get("best_val_mean_iou_6class")
                              or sd.get("best_val_miou_6class")
                              or sd.get("best_val_miou"))
        self.use_tta = use_tta
        self.encoder = encoder
        # Use train_hw from checkpoint if present (set in v6/release trainer);
        # else default 576 for MiT-family, None for CNNs.
        train_hw = sd.get("train_hw")
        if train_hw is not None:
            self.train_hw: tuple[int, int] | None = tuple(train_hw)
        elif _needs_resize(encoder):
            self.train_hw = (576, 576) if encoder.startswith("mit_") else (512, 512)
        else:
            self.train_hw = None

    @torch.no_grad()
    def predict(self, ms: np.ndarray) -> dict:
        """ms: (5, H, W) at load_ms_tile scale (~0..256). Returns numpy arrays at HxW."""
        x = torch.from_numpy(ms).float().unsqueeze(0).to(self.device)
        H, W = x.shape[-2], x.shape[-1]
        if self.train_hw is not None and (H, W) != self.train_hw:
            x_in = F.interpolate(x, size=self.train_hw, mode="bilinear", align_corners=False)
        else:
            x_in = x
        x_norm = (x_in / 256.0).clamp(0, 1)
        out = tta4_predict(self.model, x_norm) if self.use_tta else self.model(x_norm)
        if self.train_hw is not None and (H, W) != self.train_hw:
            out["seg"] = F.interpolate(out["seg"], size=(H, W), mode="bilinear", align_corners=False)
            out["ndvi"] = F.interpolate(out["ndvi"], size=(H, W), mode="bilinear", align_corners=False)
            out["ndre"] = F.interpolate(out["ndre"], size=(H, W), mode="bilinear", align_corners=False)
        return {
            "seg": out["seg"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8),
            "seg_logits": out["seg"].squeeze(0).cpu().numpy(),
            "ndvi_pred": out["ndvi"].squeeze().cpu().numpy(),
            "ndre_pred": out["ndre"].squeeze().cpu().numpy(),
            "canopy_pred": float(out["canopy"].squeeze().cpu().item()),
        }
