"""SMP DeepLabv3+ ResNet50 backbone wrapper for v3.

Replicates the WACV 2025 baseline configuration (Celikkan et al., DeepLabv3+
ResNet50, 5-channel multispectral input at native 600×600). SMP's
`DeepLabV3Plus(in_channels=5, encoder_weights='imagenet')` automatically
extends the first conv from 3 to 5 channels by averaging the ImageNet RGB
filters across the new bands — a much better init for Red Edge / NIR than
random.

Crucial: pass `encoder_output_stride=8` (not the default 16). 600 isn't
divisible by 16, so the default OS=16 decoder fails on a size mismatch
between ASPP and high-res branches. OS=8 keeps deep features at 75×75 and
costs ~30% more compute — fine for 26.7 M params.

The wrapper exposes `encoder`, `decoder`, and `segmentation_head` separately
so MultiHeadV3 can attach NDVI/NDRE heads to the decoder feature map and a
canopy-cover head to the deepest encoder feature.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def make_smp_deeplab(num_classes: int = 6, in_channels: int = 5,
                     encoder_name: str = "resnet50") -> nn.Module:
    """Build DeepLabV3Plus with the right output_stride for the chosen encoder.

    encoder_output_stride=8 only works for dilated-CNN encoders (ResNet/ResNeXt).
    Hierarchical transformers (MiT/Swin) and ConvNeXt use output_stride=16 or 32
    by default; passing 8 to them errors out.
    """
    is_dilated_cnn = (encoder_name.startswith(("resnet", "resnext", "tu-resnet",
                                               "tu-resnext", "se_resnet")))
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )
    if is_dilated_cnn:
        kwargs["encoder_output_stride"] = 8
    return smp.DeepLabV3Plus(**kwargs)


def encoder_deepest_channels(model: nn.Module) -> int:
    return model.encoder.out_channels[-1]


def decoder_out_channels(model: nn.Module) -> int:
    return model.segmentation_head[0].in_channels
