"""AgriMTL (Agricultural Multi-Task Learning) - FastAPI backend.

Wraps the existing V3Predictor + Ollama bridge in a minimal HTTP API so the
Vite/React UI can drive the same pipeline as the legacy Gradio app
(`scripts/app.py`). All numerical work is delegated to `src/` - this file is
glue.

Endpoints:
    GET  /api/state          checkpoint name, encoder, headline metric
    GET  /api/samples        canonical test-split tile ids
    GET  /api/metrics        eval_metrics_extended.json (passthrough)
    GET  /api/architecture   architecture markdown blob
    POST /api/predict        run the pipeline on a tile id
    POST /api/llm/ask        ask Ollama a free-form question with visual context
    GET  /static/<file>      colorized PNGs + GeoTIFF outputs

Run:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.inference.anomaly import anomaly_mask
from src.inference.georef import write_prescription
from src.inference.pipeline import load_ms_tile, visual_report_from
from src.inference.pipeline_v3 import V3Predictor
from src.inference.task_outputs import (colorize_stress, colorize_zones,
                                          compute_all)
from src.llm.ollama_llm import OllamaLLM


# ---- paths ------------------------------------------------------------------
CKPT_PATH = REPO / "runs" / "cropmsllm_vit.pt"
EVAL_METRICS_PATH = REPO / "runs" / "eval_metrics_extended.json"
EVAL_FALLBACK_PATH = REPO / "runs" / "eval_metrics.json"
SEVERITY_METRICS_PATH = REPO / "runs" / "severity_metrics.json"
BENCHMARKS_PATH = REPO / "runs" / "benchmarks.json"
DATASET_ROOT = REPO.parent / "data" / "weedsgalore" / "weedsgalore-dataset"
TILE_DIR_TEMPLATE = "{date}/images/{tile_id}_R.png"
STATIC_DIR = REPO / "runs" / "ui_outputs"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# ---- severity (NDVI-binned, 3-class) ---------------------------------------
# Agronomic NDVI thresholds tuned to the deployed model's operating range
# (canonical training tiles all have NDVI < 0.6, sparsely-vegetated capture
# window). Three classes: Severe (NDVI<0.2), Moderate (0.2-0.4), Mild (>=0.4).
SEVERITY_BINS = (0.2, 0.4)
SEVERITY_LABELS = ("Severe", "Moderate", "Mild")
SEVERITY_COLORS = np.array([
    [200,  50,  30],   # 0 severe   - red
    [240, 180,  50],   # 1 moderate - amber
    [ 70, 170,  70],   # 2 mild     - green
], dtype=np.uint8)


def _severity_label(ndvi: float) -> str:
    if ndvi < SEVERITY_BINS[0]: return SEVERITY_LABELS[0]
    if ndvi < SEVERITY_BINS[1]: return SEVERITY_LABELS[1]
    return SEVERITY_LABELS[2]


def severity_from_ndvi(ndvi_pred: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Bin a dense NDVI map into 3 severity categories.
    Returns (summary dict, RGB color image, per-pixel category int array)."""
    cats = np.digitize(ndvi_pred, SEVERITY_BINS).astype(np.int64)  # 0..2
    counts = np.bincount(cats.ravel(), minlength=3)
    pct = counts.astype(np.float64) / max(counts.sum(), 1) * 100.0
    dominant_idx = int(np.argmax(counts))
    image = SEVERITY_COLORS[cats]
    summary = {
        "label": SEVERITY_LABELS[dominant_idx],
        "mean_ndvi": float(ndvi_pred.mean()),
        "breakdown_pct": {
            "severe":   float(pct[0]),
            "moderate": float(pct[1]),
            "mild":     float(pct[2]),
        },
    }
    return summary, image, cats


# ---- field zoning (Safe / Attention / Critical) ----------------------------
# Aggregate per-pixel severity into a 3x3 spatial grid. Each grid cell is one
# managed zone the farmer can act on. Classification thresholds:
#   Critical (Zone C):  severe pixel share >= 30%
#   Attention (Zone B): moderate pixel share >= 30% AND not Critical
#   Safe (Zone A):      otherwise (predominantly Mild)
ZONE_LABELS = ("Safe", "Attention", "Critical")
ZONE_COLORS = np.array([
    [ 70, 170,  70],   # 0 Safe      - green
    [240, 180,  50],   # 1 Attention - amber
    [200,  50,  30],   # 2 Critical  - red
], dtype=np.uint8)
ZONE_GRID_N = 3
ZONE_CRITICAL_TH = 30.0
ZONE_ATTENTION_TH = 30.0
GRID_ROW_LABEL = ("upper", "center", "lower")
GRID_COL_LABEL = ("left", "center", "right")
ZONE_ACTIONS = {
    "Safe": "No action - vegetation health within target range.",
    "Attention": ("Monitor over the next pass. Consider a light nitrogen application "
                  "if NDRE confirms early N-deficiency."),
    "Critical": ("Immediate inspection. Targeted intervention based on dominant stress "
                 "type - irrigation if water-stress, nitrogen if N-deficiency, herbicide "
                 "if weed pressure overlaps."),
}


def compute_field_zoning(severity_cats: np.ndarray) -> tuple[dict, np.ndarray, list]:
    """3x3 spatial zoning derived from per-pixel severity.
    Returns (summary dict, RGB image (HxWx3), per-zone list)."""
    H, W = severity_cats.shape
    g = ZONE_GRID_N
    zone_class = np.zeros((g, g), dtype=np.int64)
    per_zone: list[dict] = []
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(g):
        for j in range(g):
            y0 = i * H // g;  y1 = (i + 1) * H // g
            x0 = j * W // g;  x1 = (j + 1) * W // g
            cell = severity_cats[y0:y1, x0:x1]
            n = cell.size
            pct_severe = float((cell == 0).sum()) / n * 100.0
            pct_moderate = float((cell == 1).sum()) / n * 100.0
            pct_mild = float((cell == 2).sum()) / n * 100.0
            if pct_severe >= ZONE_CRITICAL_TH:
                cls = 2
            elif pct_moderate >= ZONE_ATTENTION_TH:
                cls = 1
            else:
                cls = 0
            zone_class[i, j] = cls
            label = f"{GRID_ROW_LABEL[i]}-{GRID_COL_LABEL[j]}"
            per_zone.append({
                "row": i, "col": j,
                "cell_label": label,
                "classification": ZONE_LABELS[cls],
                "pct_severe": pct_severe,
                "pct_moderate": pct_moderate,
                "pct_mild": pct_mild,
                "action": ZONE_ACTIONS[ZONE_LABELS[cls]],
            })
            image[y0:y1, x0:x1] = ZONE_COLORS[cls]

    counts = {
        "safe":      int((zone_class == 0).sum()),
        "attention": int((zone_class == 1).sum()),
        "critical":  int((zone_class == 2).sum()),
    }
    summary = {
        "grid_n": g,
        "counts": counts,
        "total_zones": int(zone_class.size),
    }
    return summary, image, per_zone


# ---- seg palettes ----------------------------------------------------------
SEG_COLORS_6 = np.array([
    [ 50,  50,  50], [ 30, 200,  60], [220,  40,  40],
    [255, 140,   0], [255, 220,  20], [180,  80, 220],
], dtype=np.uint8)
SEG_COLORS_3 = np.array([
    [ 50,  50,  50], [ 30, 200,  60], [220,  40,  40],
], dtype=np.uint8)


def _palette_for(num_classes: int) -> np.ndarray:
    return SEG_COLORS_3 if num_classes == 3 else SEG_COLORS_6


def colorize_seg(seg: np.ndarray, num_classes: int) -> np.ndarray:
    palette = _palette_for(num_classes)
    return palette[np.clip(seg, 0, len(palette) - 1)]


def rgb_composite(ms: np.ndarray) -> np.ndarray:
    r, g, b = ms[2], ms[1], ms[0]
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = ((rgb - lo) / np.maximum(hi - lo, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def anomaly_overlay(rgb: np.ndarray, anom: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    yellow = np.array([255, 230, 0], dtype=np.uint8)
    mask = anom > 0
    if mask.any():
        out[mask] = (rgb[mask] * 0.45 + yellow * 0.55).astype(np.uint8)
    return out


def save_png(arr: np.ndarray, dest: Path) -> str:
    Image.fromarray(arr).save(dest, format="PNG")
    return f"/static/{dest.relative_to(STATIC_DIR).as_posix()}"


# ---- one-time model load ---------------------------------------------------
_state: dict = {}
_state_lock = threading.Lock()


def get_state() -> dict:
    """Idempotent model loader. Thread-safe: serialises concurrent cold-start
    requests via a lock so only the first caller pays the load cost."""
    if _state:
        return _state
    with _state_lock:
        if _state:                       # someone else loaded while we waited
            return _state
        if not CKPT_PATH.exists():
            raise RuntimeError(f"Missing checkpoint: {CKPT_PATH}")
        predictor = V3Predictor(CKPT_PATH, use_tta=True)
        llm = OllamaLLM(model="llama3.1:8b")
        _state.update(
            predictor=predictor,
            llm=llm,
            ckpt_name=CKPT_PATH.name,
            encoder=predictor.encoder,
            num_classes=predictor.num_classes,
            train_hw=predictor.train_hw,
        )
        print(f"[api] loaded {CKPT_PATH.name}  encoder={predictor.encoder}  "
              f"num_classes={predictor.num_classes}", flush=True)
        return _state


# ---- request models --------------------------------------------------------
class PredictRequest(BaseModel):
    tile_id: str
    anomaly_mode: str = "weed"


class LLMRequest(BaseModel):
    visual_report: str
    question: str
    model: str = "llama3.1:8b"


# ---- app -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Eagerly load the model + samples cache before serving any traffic."""
    get_state()
    api_samples()
    yield


app = FastAPI(title="AgriMTL API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/state")
def api_state():
    s = get_state()
    metrics = _load_metrics()
    test = (metrics or {}).get("splits", {}).get("test", {}) if metrics else {}
    return {
        "ckpt_name": s["ckpt_name"],
        "encoder": s["encoder"],
        "num_classes": s["num_classes"],
        "train_hw": list(s["train_hw"]) if s["train_hw"] else None,
        "headline_test": {
            "n_tiles": test.get("n_tiles"),
            "mean_iou_3class": test.get("mean_iou_3class"),
            "pixel_accuracy": test.get("pixel_accuracy"),
            "ndvi_mae": test.get("ndvi_mae"),
            "ndre_mae": test.get("ndre_mae"),
            "task_ndvi_rmse": test.get("task_ndvi_rmse"),
            "task_stress_accuracy": test.get("task_stress_accuracy"),
            "task_stress_macro_f1": test.get("task_stress_macro_f1"),
            "task_zone_mean_iou": test.get("task_zone_mean_iou"),
        },
    }


_samples_cache: dict | None = None


def _build_samples() -> dict:
    """Canonical test split (26 tiles) - the only data the deployed model is
    trained-distribution-valid on. Each entry carries raw-band NDVI for sort
    order; severity bucketing is intentionally omitted (it implied a per-tile
    classification verdict that pre-empts the actual model run)."""
    f = DATASET_ROOT / "splits" / "test.txt"
    if not f.exists():
        return {"tiles": [], "total": 0}
    tiles_raw = [ln.strip() for ln in open(f) if ln.strip()]
    tiles: list[dict] = []
    for tid in tiles_raw:
        date = tid[:10]
        r_path = DATASET_ROOT / date / "images" / f"{tid}_R.png"
        n_path = DATASET_ROOT / date / "images" / f"{tid}_NIR.png"
        if not r_path.exists() or not n_path.exists():
            continue
        r = np.array(Image.open(r_path), dtype=np.float32)
        if r.ndim == 3: r = r[..., 0]
        r /= 255.0
        n = np.array(Image.open(n_path), dtype=np.float32)
        if n.ndim == 3: n = n[..., 0]
        n /= 255.0
        ndvi = float(((n - r) / (n + r + 1e-6)).mean())
        tiles.append({"tile_id": tid, "ndvi": round(ndvi, 3)})
    tiles.sort(key=lambda e: e["ndvi"])
    return {"tiles": tiles, "total": len(tiles)}


@app.get("/api/samples")
def api_samples():
    global _samples_cache
    if _samples_cache is None:
        _samples_cache = _build_samples()
        print(f"[api] samples cached: total={_samples_cache['total']}",
              flush=True)
    return _samples_cache


def _load_metrics() -> dict | None:
    p = EVAL_METRICS_PATH if EVAL_METRICS_PATH.exists() else (
        EVAL_FALLBACK_PATH if EVAL_FALLBACK_PATH.exists() else None)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/api/metrics")
def api_metrics():
    m = _load_metrics()
    if m is None:
        raise HTTPException(404, "No eval metrics on disk")
    out = dict(m)
    # Drop test_balanced (out-of-distribution split, not part of the deployed
    # model's valid operating range) and the legacy Health-F1 column (replaced
    # by the 3-class severity classifier in the new implementation).
    splits = dict(out.get("splits") or {})
    splits.pop("test_balanced", None)
    cleaned: dict = {}
    for name, m_split in splits.items():
        m_split = dict(m_split)
        m_split.pop("task_health_f1", None)
        cleaned[name] = m_split
    out["splits"] = cleaned
    if SEVERITY_METRICS_PATH.exists():
        try:
            out["severity"] = json.loads(SEVERITY_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if BENCHMARKS_PATH.exists():
        try:
            out["benchmarks"] = json.loads(BENCHMARKS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out


@app.get("/api/architecture")
def api_architecture():
    return {
        "pipeline": (
            "Input MS tile (5, 600, 600)\n"
            "  -> resize 576x576, /256 -> [0, 1]\n"
            "       -> MiT-B3 encoder (4 stages, hierarchical ViT)\n"
            "             -> DeepLabv3+ decoder (ASPP + skip)\n"
            "                   -> seg head     -> (C, H, W)  argmax -> class mask\n"
            "                   -> NDVI head    -> (1, H, W)  tanh -> [-1, 1]\n"
            "                   -> NDRE head    -> (1, H, W)  tanh -> [-1, 1]\n"
            "                   -> canopy head  -> (1,)       sigmoid -> [0, 1]\n"
            "  -> upsample seg/NDVI/NDRE -> (B, *, 600, 600)\n"
            "  -> task_outputs.compute_all() -> 5 precision-ag derivations\n"
            "  -> visual_report -> Ollama LLM -> farmer-language answer\n"
            "  -> write_prescription -> 2-band GeoTIFF"
        ),
        "dataset": (
            "Labeled UAV multispectral tiles, MicaSense RedEdge-MX. "
            "5 bands (B 475 / G 560 / R 668 / RE 717 / NIR 840 nm), uint16 PNG @ 600x600. "
            "Maize crop with weed sub-species labels; severe class imbalance (bg ~83%). "
            "Trained at 576x576 input resolution; predictions upscaled to native 600x600."
        ),
        "operating_range": (
            "Trained and validated on canonical tiles with NDVI < 0.6 (sparsely-vegetated "
            "capture window). Stress severity is reported in three of four NDVI bins - "
            "Severe (<0.2), Moderate (0.2-0.4), Mild (0.4-0.6) - which fully cover the "
            "valid input distribution. The fourth bin (No stress, NDVI >= 0.6) appears "
            "in the legend for completeness but is outside the deployed model's training "
            "distribution."
        ),
    }


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    s = get_state()
    tile_id = req.tile_id
    if not tile_id:
        raise HTTPException(400, "tile_id required")
    date = tile_id[:10]
    ms_path = DATASET_ROOT / TILE_DIR_TEMPLATE.format(date=date, tile_id=tile_id)
    if not ms_path.exists():
        raise HTTPException(404, f"Tile not found: {ms_path}")

    t0 = time.time()
    ms, crs, transform = load_ms_tile(ms_path)

    # Closed-form NDVI from raw bands - exposed alongside the predicted NDVI
    # so the UI can show actual vs predicted severity.
    red = ms[2].astype(np.float32) / 255.0
    nir = ms[4].astype(np.float32) / 255.0
    ndvi_actual = (nir - red) / (nir + red + 1e-6)
    ndvi_actual_mean = float(ndvi_actual.mean())

    pred = s["predictor"].predict(ms)
    seg = pred["seg"]
    seg_logits = pred["seg_logits"]
    nc = s["num_classes"]

    anom = anomaly_mask(ms, mode=req.anomaly_mode, seg_logits=seg_logits)
    rgb = rgb_composite(ms)
    # Display-time collapse: the model is 6-class native (bg/maize/weed_a..d)
    # but the UI consistently shows 3 classes (bg / maize / merged-weed) to
    # match the headline metric. The GeoTIFF still gets the raw 6-class
    # prediction in band 1 for downstream consumers.
    seg_for_display = seg.copy()
    seg_for_display[seg_for_display > 1] = 2
    seg_color = colorize_seg(seg_for_display, num_classes=3)
    anom_overlay_img = anomaly_overlay(rgb, anom)

    tasks = compute_all(ms, seg, pred["ndvi_pred"], pred["ndre_pred"])
    health_rgb = tasks.health_intensity_rgb
    stress_rgb = colorize_stress(tasks.stress_type_map)
    severity_summary, severity_rgb, severity_cats = severity_from_ndvi(pred["ndvi_pred"])
    severity_actual_summary, _, _ = severity_from_ndvi(ndvi_actual)
    zoning_summary, zoning_rgb, per_zone = compute_field_zoning(severity_cats)

    visual_report = visual_report_from(seg, anom)
    bd = severity_summary["breakdown_pct"]
    zc = zoning_summary["counts"]
    critical_zones = [z for z in per_zone if z["classification"] == "Critical"]
    attention_zones = [z for z in per_zone if z["classification"] == "Attention"]
    actionable_text = ""
    if critical_zones:
        actionable_text += (
            "\nCritical zones (immediate intervention):\n"
            + "\n".join(f"  - {z['cell_label']}  severe {z['pct_severe']:.0f}%, "
                        f"moderate {z['pct_moderate']:.0f}%" for z in critical_zones)
        )
    if attention_zones:
        actionable_text += (
            "\nAttention zones (monitor):\n"
            + "\n".join(f"  - {z['cell_label']}  moderate {z['pct_moderate']:.0f}%, "
                        f"severe {z['pct_severe']:.0f}%" for z in attention_zones)
        )
    visual_report = (
        f"{visual_report}\n\n"
        f"Stress severity (3-class NDVI bins): dominant {severity_summary['label']} "
        f"(predicted mean NDVI {severity_summary['mean_ndvi']:+.3f}; "
        f"raw-band {ndvi_actual_mean:+.3f}).\n"
        f"  Severe   {bd['severe']:.1f}%   (NDVI < {SEVERITY_BINS[0]:.1f})\n"
        f"  Moderate {bd['moderate']:.1f}% ({SEVERITY_BINS[0]:.1f}-{SEVERITY_BINS[1]:.1f})\n"
        f"  Mild     {bd['mild']:.1f}%     (NDVI >= {SEVERITY_BINS[1]:.1f})\n"
        f"Dominant stress type: {tasks.stress_type}\n"
        f"Field zoning (3x3 grid, {zoning_summary['total_zones']} zones): "
        f"{zc['critical']} Critical · {zc['attention']} Attention · {zc['safe']} Safe."
        f"{actionable_text}"
    )

    request_id = uuid.uuid4().hex[:8]
    req_dir = STATIC_DIR / request_id
    req_dir.mkdir(parents=True, exist_ok=True)

    rgb_url = save_png(rgb, req_dir / "rgb.png")
    seg_url = save_png(seg_color, req_dir / "seg.png")
    anom_url = save_png(anom_overlay_img, req_dir / "anomaly.png")
    health_url = save_png(health_rgb, req_dir / "health.png")
    stress_url = save_png(stress_rgb, req_dir / "stress.png")
    severity_url = save_png(severity_rgb, req_dir / "severity.png")
    zoning_url = save_png(zoning_rgb, req_dir / "zoning_abc.png")

    out_tif = req_dir / f"{ms_path.stem.rsplit('_', 1)[0]}_prescription.tif"
    write_prescription(out_tif, seg, None, anom, crs, transform)
    tif_url = f"/static/{out_tif.relative_to(STATIC_DIR).as_posix()}"

    # Render the GeoTIFF preview the same way the Gradio app did.
    try:
        import rasterio
        with rasterio.open(out_tif) as ds:
            seg_band = ds.read(1)
            anom_band = ds.read(2) if ds.count >= 2 else None
        preview = colorize_seg(seg_band, num_classes=nc).copy()
        if anom_band is not None and (anom_band > 0).any():
            yellow = np.array([255, 230, 0], dtype=np.uint8)
            mask = anom_band > 0
            preview[mask] = (preview[mask] * 0.55 + yellow * 0.45).astype(np.uint8)
    except Exception:
        preview = seg_color.copy()
    preview_url = save_png(preview, req_dir / "prescription_preview.png")

    elapsed = time.time() - t0

    return {
        "request_id": request_id,
        "tile_id": tile_id,
        "elapsed_s": round(elapsed, 2),
        "anomaly_mode": req.anomaly_mode,
        "num_classes": nc,
        "images": {
            "rgb": rgb_url,
            "seg": seg_url,
            "anomaly": anom_url,
            "severity": severity_url,
            "zoning": zoning_url,
            "health": health_url,
            "stress": stress_url,
            "prescription_preview": preview_url,
        },
        "severity": {
            **severity_summary,
            "actual_label": severity_actual_summary["label"],
            "actual_mean_ndvi": ndvi_actual_mean,
        },
        "zoning": {
            **zoning_summary,
            "zones": per_zone,
        },
        "tasks": {
            "stress_type": tasks.stress_type,
            "spatial_text": tasks.spatial_text,
        },
        "canopy_pred": float(pred["canopy_pred"]),
        "visual_report": visual_report,
        "geotiff_url": tif_url,
    }


@app.post("/api/llm/ask")
def api_llm_ask(req: LLMRequest):
    s = get_state()
    if req.model and req.model != s["llm"].model:
        s["llm"] = OllamaLLM(model=req.model)
    if not s["llm"].health():
        raise HTTPException(503, f"Ollama model `{s['llm'].model}` not running. "
                                  f"Start `ollama serve` and `ollama pull {s['llm'].model}`.")
    try:
        out = s["llm"].generate(
            visual_report=req.visual_report, rag_context="", farmer_query=req.question)
        return {"model": s["llm"].model, "answer": out.text}
    except Exception as e:
        raise HTTPException(500, f"Ollama call failed: {e}")
