# CropMS-LLM ViT — Release

Self-contained ViT-only deployment of the WeedsGalore multi-task pipeline.
Single canonical model, one train script, one eval script, FastAPI backend +
React UI, optional Ollama LLM bridge.

**For setup, follow [RUNBOOK.md](RUNBOOK.md).** TL;DR: pyenv → venv → pip install →
`jupyter lab notebooks/full_pipeline.ipynb`. CPU works; GPU is a 5–30× speedup.

## Architecture

```
Input: 5-band MS UAV tile (B, G, R, RE, NIR) — uint16 PNGs at 600×600
       │
       ▼
┌───────────────────────────────────────┐
│  MiT-B3 encoder (SegFormer ViT)       │
│  • Hierarchical transformer (4 stages)│
│  • ImageNet pretrained                │
│  • conv1 extended 3→5 channels        │
│  • 45.2M parameters                   │
└─────────────┬─────────────────────────┘
              │ multi-scale features
              ▼
┌───────────────────────────────────────┐
│  DeepLabv3+ decoder (256-channel ASPP)│
└──┬───────┬──────────┬──────────┬──────┘
   │       │          │          │
   ▼       ▼          ▼          ▼
 ┌────┐ ┌──────┐ ┌──────┐ ┌──────────┐
 │seg │ │NDVI  │ │NDRE  │ │canopy    │
 │6cls│ │dense │ │dense │ │scalar    │
 └────┘ └──────┘ └──────┘ └──────────┘
   │
   ▼ (4-flip TTA averaged, all upsampled to 600×600)

Loss = KendallUncertainty(
         dice + classwt-focalCE + 0.5·Lovász,    ← seg
         L1(NDVI), L1(NDRE), L1(canopy)          ← spectral aux heads
       )
```

**Why ViT?** Hierarchical self-attention captures field-wide spatial relationships
that CNNs miss. Multi-spectral fusion via the 5-channel first conv lets the
transformer learn cross-band stress signatures (esp. Red Edge ↔ NIR for nitrogen
deficiency).

**Recipe** (per `scripts/train.py`):
- Train res 576×576 (multiple of 32 required by MiT)
- AdamW, weight decay 1e-4
- 5 warmup epochs (encoder frozen, lr 1e-3) → 35 main epochs (full unfreeze, lr 3e-4 cosine to 1e-6 with 5% linear warmup)
- EMA decay 0.99 with 50-step verbatim warmup
- Augmentation: H/V flip + rot90 + weed-instance copy-paste (p=0.5) + MS-aware mixup (p=0.3, λ_max=0.3)
- Class weights: sqrt-inverse-frequency
- Mixed precision (fp16 autocast)
- Peak VRAM ≈ 6.4 GB on RTX 4060 8 GB

## Performance

Test split (26 tiles), 4-flip TTA:

| Metric | Score | vs WACV 2025 baseline |
|---|---:|---:|
| **3-class mIoU** (bg/crop/merged-weed) | **81.40%** | −1.50 pp |
| 6-class mIoU (separate weed_a/b/c/d) | 56.53% | — |
| Pixel accuracy | 97.86% | — |
| NDVI MAE (dense, [-1, 1]) | 0.046 | — |
| NDRE MAE (dense, [-1, 1]) | 0.034 | — |

The 3-class metric is the WACV 2025 paper headline (their `evaluate.py num_classes=3`).
The 6-class metric is harder because rare weed_d (~50 px/tile) drags the average.

## Quick start

Full step-by-step in [RUNBOOK.md](RUNBOOK.md). Condensed:

```bash
pyenv install 3.10.14 && pyenv local 3.10.14
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel

# CPU PyTorch (any OS, no GPU needed):
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.3.*" torchvision
# OR GPU (CUDA 12.1):
# pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.*" torchvision

pip install -r requirements.txt

# 1) Notebook walkthrough (inference + eval + training, on bundled sample tiles)
jupyter lab notebooks/full_pipeline.ipynb

# 2) Or run the API + UI:
uvicorn api.main:app --port 8001        # backend:  http://localhost:8001
cd ui && npm install && npm run dev     # frontend: http://localhost:5173
```

The pretrained checkpoint ships at `runs/cropmsllm_vit.pt` and 5 sample tiles
ship at `data/weedsgalore/weedsgalore-dataset/` so the notebook and UI work
immediately without downloading WeedsGalore.

## UI tour

The React UI (in `ui/`) talks to the FastAPI backend (`api/main.py`) and exposes:

- **RGB composite** (percentile-stretched R/G/B view)
- **Seg mask** (6-class colorized)
- **Anomaly overlay** (configurable mode: weed / stress / uncertainty / union)
- **5 precision-ag derivations**:
  1. Crop Health label (Healthy / Stressed; NDVI ≥ 0.6 threshold)
  2. NDVI / Health intensity map (red→yellow→green colormap)
  3. Stress type per pixel (N-def / Water / General; spectral heuristic)
  4. Field zoning (good / poor / bare)
  5. 3×3 spatial grid summary
- **Ollama LLM** Q&A panel (optional; needs `ollama pull llama3.1:8b`)
- **Prescription GeoTIFF** auto-saved to `runs/ui_outputs/<tile>_prescription.tif`

## Directory layout

```
.
├── README.md                       ← this file
├── RUNBOOK.md                      ← step-by-step setup
├── .python-version                 ← pyenv pin (3.10.14)
├── requirements.txt                ← pip deps (torch installed separately)
├── api/
│   └── main.py                     ← FastAPI backend (port 8001)
├── src/
│   ├── data/weedsgalore.py         ← 5-band MS loader (per-band uint16 PNG)
│   ├── data/augment_v3.py          ← copy-paste + MS-aware mixup
│   ├── models/                     ← backbone, multi-head, losses
│   ├── inference/
│   │   ├── pipeline.py             ← load_ms_tile, visual_report_from
│   │   ├── pipeline_v3.py          ← V3Predictor (single-tile inference)
│   │   ├── tta.py                  ← 4-flip averaged forward
│   │   ├── task_outputs.py         ← 5 task derivations + colorize_*
│   │   ├── anomaly.py              ← weed/stress anomaly mask
│   │   └── georef.py               ← GeoTIFF prescription writer
│   └── llm/ollama_llm.py           ← Ollama HTTP client + prompt builder
├── scripts/
│   ├── train.py                    ← canonical training entry point
│   └── eval.py                     ← train/val/test metrics with TTA
├── ui/                             ← React + Vite frontend (port 5173)
├── notebooks/
│   └── full_pipeline.ipynb         ← inference + eval + training walkthrough
├── data/weedsgalore/weedsgalore-dataset/
│   ├── 2023-{05-25,05-30,06-06,06-15}/  ← 5 bundled sample tiles (5 bands + masks)
│   └── splits/{train,val,test}.txt      ← bundled splits (test = 5 tiles)
└── runs/
    ├── cropmsllm_vit.pt            ← pretrained checkpoint (45.2M params, 175 MB)
    ├── cfg.json                    ← training config used
    ├── eval_metrics_extended.json  ← held-out metrics (UI consumes this)
    ├── benchmarks.json
    └── severity_metrics.json
```

## Citing

Built on:
- WeedsGalore dataset, Celikkan et al., WACV 2025 ([arXiv:2502.13103](https://arxiv.org/abs/2502.13103))
- SegFormer/MiT, Xie et al., NeurIPS 2021 ([arXiv:2105.15203](https://arxiv.org/abs/2105.15203))
- DeepLabv3+, Chen et al., ECCV 2018
- Lovász-Softmax, Berman et al., CVPR 2018
- Multi-task uncertainty, Kendall & Gal, CVPR 2018
