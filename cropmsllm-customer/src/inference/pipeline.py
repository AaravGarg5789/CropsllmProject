"""End-to-end inference: tile + farmer query -> text answer + prescription GeoTIFF.

Architecture (Ollama-backed deployment):

  1. Load 5-band MSI tile (PNG-per-band per WeedsGalore on-disk layout, or a
     stacked GeoTIFF if available).
  2. BAMT-v2 -> ViT backbone -> UPerNet seg head  → per-pixel class logits
                                                    (frozen, from runs/stage1.pt)
  3. Q-Former  → 32 LLM-space tokens reserved for the future fine-tunable LLM
                 path. Not used by the Ollama call.
  4. Spatial Anomaly Module (weed | stress | uncertainty | union)
                                                  → per-pixel anomaly mask
  5. Build a structured `visual_report` string from (a) class composition,
     (b) where-the-weeds-are summary, (c) anomaly hotspot summary.
  6. Chroma-RAG retrieval against the agronomic KB (optional, gracefully
     skipped if KB is empty).
  7. Ollama (llama3.1:8b by default) generates the farmer-facing answer.
  8. Write a 2-band georeferenced GeoTIFF: seg classes + anomaly mask.
     CRS / transform are preserved if the input tile carried them; otherwise
     written as `EPSG:4326` with an identity transform (placeholder for
     non-georeferenced PNG-per-band tiles like WeedsGalore).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.inference.anomaly import anomaly_mask
from src.inference.georef import write_prescription


CLASS_NAMES = {0: "background", 1: "maize crop", 2: "weed", 3: "weed", 4: "weed", 5: "weed"}


def load_ms_tile(ms_path: Path) -> tuple[np.ndarray, str, tuple]:
    """Load a 5-band MSI tile.

    Two layouts supported:
      A) A single multi-band GeoTIFF (rasterio).
      B) WeedsGalore PNG-per-band: pass any of the per-band PNG paths and the
         loader will discover its siblings (B/G/R/RE/NIR).
    """
    p = ms_path
    suffix = p.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(p) as ds:
            ms = ds.read().astype(np.float32) / 255.0
            crs = str(ds.crs) if ds.crs else "EPSG:4326"
            transform = tuple(ds.transform)
        return ms, crs, transform

    # PNG-per-band layout
    from PIL import Image
    base = p.parent / p.stem.rsplit("_", 1)[0]
    bands = []
    for suf in ("B", "G", "R", "RE", "NIR"):
        bp = Path(f"{base}_{suf}.png")
        arr = np.array(Image.open(bp), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        bands.append(arr)
    ms = np.stack(bands, axis=0) / 255.0
    return ms, "EPSG:4326", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def visual_report_from(seg_classes: np.ndarray, anomaly: np.ndarray) -> str:
    """Build a compact structured caption for the LLM."""
    H, W = seg_classes.shape
    flat = seg_classes.flatten()
    pct = np.bincount(flat, minlength=6).astype(np.float64) / max(flat.size, 1)
    bg = float(pct[0])
    crop = float(pct[1])
    weed = float(pct[2:].sum())

    if H >= W:
        n_half, s_half = seg_classes[: H // 2], seg_classes[H // 2 :]
        north_label, south_label = "northern", "southern"
    else:
        n_half, s_half = seg_classes[:, : W // 2], seg_classes[:, W // 2 :]
        north_label, south_label = "western", "eastern"
    weed_n = float(((n_half >= 2) & (n_half < 6)).mean())
    weed_s = float(((s_half >= 2) & (s_half < 6)).mean())
    region = north_label if weed_n >= weed_s else south_label
    weed_pct_in_region = max(weed_n, weed_s) * 100

    anom_pct = float(anomaly.mean()) * 100

    lines = [
        f"- Class composition: {bg * 100:.1f}% bare soil, {crop * 100:.1f}% maize crop, {weed * 100:.1f}% weeds.",
        f"- Weed concentration: heaviest in the {region} half ({weed_pct_in_region:.1f}% weed cover there).",
        f"- Anomaly mask coverage: {anom_pct:.1f}% of the tile flagged for inspection.",
    ]
    return "\n".join(lines)


def run(
    ms_path: Path,
    query: str,
    out_path: Path,
    stage1_ckpt: Path,
    stage2_ckpt: Optional[Path] = None,
    anomaly_mode: str = "weed",
    ollama_model: str = "llama3.1:8b",
    use_rag: bool = False,
) -> str:
    from src.models.bamt import BAMTv2
    from src.models.backbone_prithvi import PrithviBackbone
    from src.models.qformer import QFormer
    from src.models.seg_head import UPerNetMS
    from src.llm.ollama_llm import OllamaLLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ms, crs, transform = load_ms_tile(ms_path)
    H, W = ms.shape[1], ms.shape[2]
    ms_t = torch.from_numpy(ms).unsqueeze(0).float().to(device)

    # Build models, load stage-1 weights
    bamt = BAMTv2(num_bands=ms.shape[0], out_channels=64).to(device)
    backbone = PrithviBackbone(in_bands=64, pretrained=True, lora_rank=0).to(device)
    seg_head = UPerNetMS(in_channels=[768], num_classes=6).to(device)
    sd1 = torch.load(stage1_ckpt, map_location=device)
    bamt.load_state_dict(sd1["bamt"])
    backbone.load_state_dict(sd1["backbone"])
    seg_head.load_state_dict(sd1["head"])
    bamt.eval(); backbone.eval(); seg_head.eval()

    qformer = QFormer().to(device).eval()
    if stage2_ckpt is not None and stage2_ckpt.exists():
        sd2 = torch.load(stage2_ckpt, map_location=device)
        qformer.load_state_dict(sd2["qformer"])

    # Resize input to 224 for the ViT backbone (matches Stage 1 training)
    import torch.nn.functional as F
    ms_224 = F.interpolate(ms_t, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        feats = backbone(bamt(ms_224))
        seg_logits = seg_head(feats["features"], output_size=(H, W))
    seg_classes = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    seg_logits_np = seg_logits.squeeze(0).cpu().numpy()

    anom = anomaly_mask(ms, mode=anomaly_mode, seg_logits=seg_logits_np)
    write_prescription(out_path, seg_classes, None, anom, crs, transform)

    # Visual report → LLM
    visual_report = visual_report_from(seg_classes, anom)
    rag_context = ""
    if use_rag:
        try:
            from src.llm.rag import AgroKB
            kb = AgroKB()
            retrieved = kb.retrieve(query, k=3)
            rag_context = AgroKB.format_context(retrieved) if retrieved else ""
        except Exception as e:
            print(f"[warn] RAG disabled: {e}")

    llm = OllamaLLM(model=ollama_model)
    if not llm.health():
        return f"[error] Ollama model {ollama_model} not available; visual report:\n{visual_report}"
    out = llm.generate(visual_report=visual_report, rag_context=rag_context, farmer_query=query)
    return out.text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=Path, required=True, help="5-band MSI tile (GeoTIFF or any per-band PNG)")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("prescription.tif"))
    ap.add_argument("--stage1-ckpt", type=Path, default=Path("runs/stage1.pt"))
    ap.add_argument("--stage2-ckpt", type=Path, default=Path("runs/stage2.pt"))
    ap.add_argument("--anomaly-mode", choices=["weed", "stress", "uncertainty", "union"], default="weed")
    ap.add_argument("--ollama-model", default="llama3.1:8b")
    ap.add_argument("--rag", action="store_true", help="enable Chroma-RAG over the agronomic KB")
    args = ap.parse_args()
    answer = run(
        args.ms, args.query, args.out,
        stage1_ckpt=args.stage1_ckpt, stage2_ckpt=args.stage2_ckpt,
        anomaly_mode=args.anomaly_mode, ollama_model=args.ollama_model, use_rag=args.rag,
    )
    print("=" * 80)
    print("ANSWER:")
    print(answer)
    print("=" * 80)
    print(f"PRESCRIPTION MAP: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
