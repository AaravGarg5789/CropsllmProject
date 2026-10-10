"""WeedsGalore dataset loader.

Verified on-disk layout (after running scripts/download_weedsgalore.py):

    weedsgalore-dataset/
        <YYYY-MM-DD>/
            images/
                <id>_B.png       # Blue
                <id>_G.png       # Green
                <id>_R.png       # Red
                <id>_RE.png      # Red-Edge
                <id>_NIR.png     # Near-Infrared
            semantics/<id>.png   # uint8 class indices
            instances/<id>.png   # instance ids (unused for semantic seg)
            logs/...
        ...
        splits/{train,val,test}.txt
        LICENSE.txt

Each split file lists tile IDs in the form `<date>_<num>` (e.g.,
`2023-06-15_0472`). The loader resolves them to the per-date subfolder and
stacks the 5 single-band PNGs into a (5, H, W) MSI tensor in band order
[Blue, Green, Red, Red-Edge, NIR].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


BAND_SUFFIXES = ("B", "G", "R", "RE", "NIR")


@dataclass
class Tile:
    ms: torch.Tensor                # (5, H, W) float32
    mask: torch.Tensor              # (H, W) int64
    crs: Optional[str]              # WeedsGalore PNG tiles carry no CRS
    transform: Optional[tuple]
    tile_id: str                    # "<date>_<num>"


class WeedsGalore(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        ms_norm: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.ms_norm = ms_norm
        with open(self.root / "splits" / f"{split}.txt") as fh:
            self.ids = [line.strip() for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _parse_id(tid: str) -> tuple[str, str]:
        """Resolve <date>_<num>-style tile IDs.

        Two id schemes coexist on disk:
          - labeled tiles: `2023-06-15_0472`            -> ('2023-06-15', '0472')
          - ortho-pseudo: `2023-06-15_om_000672_013888` -> ('2023-06-15', '...')
        """
        if "_om_" in tid:
            date = tid.split("_om_", 1)[0]
            return date, tid                            # full tid is the file basename
        idx = tid.rfind("_")
        if idx == -1:
            raise ValueError(f"unrecognized tile id format: {tid}")
        return tid[:idx], tid[idx + 1 :]

    def _resolve(self, tid: str) -> tuple[list[Path], Path]:
        date, base = self._parse_id(tid)
        bands = [self.root / date / "images" / f"{tid}_{suffix}.png" for suffix in BAND_SUFFIXES]
        mask = self.root / date / "semantics" / f"{tid}.png"
        return bands, mask

    def __getitem__(self, idx: int) -> Tile:
        from PIL import Image

        tid = self.ids[idx]
        band_paths, mask_path = self._resolve(tid)

        bands = []
        for p in band_paths:
            arr = np.array(Image.open(p), dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]  # collapse accidental RGB-encoded single-band PNGs
            bands.append(arr)
        ms = np.stack(bands, axis=0) / 255.0                  # (5, H, W) normalized to [0, 1]
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if self.ms_norm is not None:
            mean, std = self.ms_norm
            ms = (ms - mean[:, None, None]) / std[:, None, None]

        if self.transform is not None:
            ms, mask = self.transform(ms, mask)

        return Tile(
            ms=torch.from_numpy(np.ascontiguousarray(ms)).float(),
            mask=torch.from_numpy(np.ascontiguousarray(mask)).long(),
            crs=None,
            transform=None,
            tile_id=tid,
        )
