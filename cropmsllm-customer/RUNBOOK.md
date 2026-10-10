# RUNBOOK — replicate locally

Step-by-step setup for running the shipped CropMS-LLM ViT pipeline on your own
machine. **No Docker required.** GPU is optional — inference runs on CPU.

What you'll get at the end:
- A working Python venv with the shipped checkpoint loadable
- The Jupyter notebook (`notebooks/full_pipeline.ipynb`) running inference on bundled sample tiles
- The FastAPI backend on `http://localhost:8001` and the React UI on `http://localhost:5173`
- Optional: Ollama LLM bridge for the Q&A panel in the UI

---

## 1. Prerequisites

| | macOS | Linux | Windows |
|---|---|---|---|
| **pyenv** | `brew install pyenv` | [pyenv installer](https://github.com/pyenv/pyenv#installation) | `pyenv-win` (different tool, see notes) |
| **Node.js 20+** (UI only) | `brew install node@20` | [nodesource](https://github.com/nodesource/distributions) | [nodejs.org](https://nodejs.org/) |
| **Ollama** (LLM only, optional) | [ollama.com/download](https://ollama.com/download) | same | same |

For Windows users: `pyenv-win` works but path handling differs. If you hit issues,
use WSL2 + the Linux instructions instead — that's our supported path on Windows.

---

## 2. Python environment (pyenv + venv)

```bash
# from the project root (where this RUNBOOK lives)

# 1) Pin the Python version (.python-version is committed)
pyenv install 3.10.14    # one-time; downloads & builds CPython
pyenv local 3.10.14      # picks up .python-version

# 2) Create + activate a venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python -V                   # should print 3.10.14
pip install --upgrade pip wheel
```

---

## 3. Install PyTorch (pick CPU **or** GPU)

PyTorch is the one dep where the wheel choice matters. Install it **first**, then
the rest of `requirements.txt` will skip it.

### CPU-only (any OS, no GPU required)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.3.*" torchvision
```

### GPU — NVIDIA CUDA 12.1 (Linux / Windows with recent driver)

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.*" torchvision
```

Verify:
```bash
python -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
```

---

## 4. Install the rest of the deps

```bash
pip install -r requirements.txt
```

This pulls FastAPI, transformers, segmentation-models-pytorch, rasterio, jupyterlab,
matplotlib, etc. (~600 MB on disk for CPU; ~3 GB if you went GPU above.)

---

## 5. Run the notebook

```bash
jupyter lab notebooks/full_pipeline.ipynb
```

The notebook walks through:
1. Loading `runs/cropmsllm_vit.pt` (the shipped checkpoint)
2. Running inference on the 5 bundled sample tiles (in `data/weedsgalore/weedsgalore-dataset/`)
3. Visualizing all 6 outputs the UI shows (RGB, seg mask, NDVI, NDRE, stress, zones)
4. Reading the held-out test metrics (3-class mIoU 81.4%, etc.)
5. Optional: smoke-training cell + full retrain command

CPU inference per tile is ~10–30 s (4-flip TTA × MiT-B3). On an 8 GB GPU it's ~0.5 s.

---

## 6. Run the API + UI (UI-equivalent of the notebook)

### 6a. Backend

```bash
# from the project root, venv active
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

Sanity-check:
```bash
curl http://localhost:8001/api/state
# => {"ckpt_name":"cropmsllm_vit.pt","encoder":"mit_b3", ...}
```

### 6b. Frontend

```bash
cd ui
npm install      # one-time, ~30 s
npm run dev      # vite dev server
# open http://localhost:5173
```

`vite.config.ts` already proxies `/api/*` and `/static/*` to `127.0.0.1:8001`, so
the UI and API just work together. For production hosting, run `npm run build`
and serve `ui/dist/` from any static host (nginx, Caddy, S3+CloudFront).

---

## 7. Optional — LLM Q&A panel via Ollama

The UI has an "Ask the assistant" panel that calls `/api/llm/ask`. It needs a
local Ollama daemon. Skip this section if you don't need the LLM feature — every
other page works without it.

```bash
# install Ollama (one-time, see ollama.com/download)
ollama serve &                  # default: http://localhost:11434

# pull the model the API expects
ollama pull llama3.1:8b         # ~5 GB; runs on CPU but is slow without a GPU

# OR pick a smaller CPU-friendly model and tell the API to use it:
ollama pull phi3:3.8b
OLLAMA_MODEL=phi3:3.8b uvicorn api.main:app --host 0.0.0.0 --port 8001
```

**Env vars the API honors:**
- `OLLAMA_URL` — default `http://localhost:11434`
- `OLLAMA_MODEL` — default `llama3.1:8b`

---

## 8. Re-run eval / retrain

Eval against the bundled 5 tiles:
```bash
python scripts/eval.py --splits test
```

Full retrain (needs the full WeedsGalore dataset; the ship contains only 5 tiles):
```bash
# 1) download WeedsGalore from arXiv:2502.13103 supplementary materials
# 2) place under data/weedsgalore/weedsgalore-dataset/
# 3) restore the original splits files (train.txt / val.txt / test.txt)
# 4) train
python scripts/train.py
# -> writes runs/cropmsllm_vit.pt and runs/<encoder>.log.json
```

Recipe (already encoded in `scripts/train.py` defaults): MiT-B3 + DeepLabv3+
decoder, 5 warmup + 35 main epochs, AdamW with cosine LR, multi-task Kendall
uncertainty loss, mixed precision. ~25 min on an RTX 4060 8 GB.

---

## Troubleshooting

**`ImportError: cannot import name '_C' from 'torch'`** — torch wheel mismatch with
your Python version. Re-create the venv with Python 3.10 specifically.

**`OSError: [E050] could not load rasterio`** — rare on macOS. Try `pip install
--no-binary rasterio rasterio` (compiles against system GDAL via `brew install gdal`).

**API returns 503 on `/api/llm/ask`** — Ollama is not running, or the model
referenced (`llama3.1:8b` by default) is not pulled. See section 7.

**Notebook can't import `src.…`** — the first setup cell fixes `sys.path` for you.
If you launched Jupyter from somewhere other than the project root, restart the
kernel and run all cells from the top.

**CPU inference is too slow** — drop TTA: `V3Predictor(CKPT, use_tta=False)` is
~4× faster (single forward pass instead of 4-flip averaging) at a small accuracy
cost (~0.5 pp mIoU).
