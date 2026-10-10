"""Ollama-backed LLM client for the inference pipeline.

Replaces the original `rice_llm.py` (HF Llama + QLoRA + LISA <SEG>) for the
8 GB-VRAM laptop deployment. Trade-offs vs. the HF route:

  + No HF gating, no QLoRA fine-tune, no instruction-data fabrication
  + 4-bit quantization handled by Ollama; ~5 GB VRAM for llama3.1:8b
  - No <SEG> token grounding (Ollama exposes no hidden states); the LLM
    cannot emit an LLM-conditioned mask. Spatial output comes from the seg
    head + anomaly module only.
  - No visual-prefix tokens; the visual side is summarized into structured
    text (seg-mask caption) before prompting the LLM.

The original `rice_llm.py` is retained for the future fine-tunable path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import requests


DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


@dataclass
class OllamaResponse:
    text: str
    model: str
    eval_count: int
    prompt_eval_count: int


SYSTEM_PROMPT = (
    "You are an agricultural decision-support assistant for farmers operating"
    " UAVs over crop fields. You receive a structured visual report (class"
    " composition + spatial anomaly summary) plus retrieved agronomic"
    " context, then a farmer's natural-language question. Answer in 2-4"
    " short sentences with concrete, actionable guidance. Never invent"
    " quantitative numbers; if the visual report doesn't say something, do"
    " not claim it. If the visual report indicates the field is healthy,"
    " say so plainly."
)


class OllamaLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_URL,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
            names = {m["name"] for m in r.json().get("models", [])}
            return self.model in names
        except Exception:
            return False

    def generate(
        self,
        visual_report: str,
        rag_context: str,
        farmer_query: str,
        temperature: float = 0.2,
    ) -> OllamaResponse:
        prompt = self._build_prompt(visual_report, rag_context, farmer_query)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": 4096},
        }
        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return OllamaResponse(
            text=data.get("response", "").strip(),
            model=self.model,
            eval_count=int(data.get("eval_count", 0)),
            prompt_eval_count=int(data.get("prompt_eval_count", 0)),
        )

    @staticmethod
    def _build_prompt(visual_report: str, rag_context: str, farmer_query: str) -> str:
        rag_block = f"\n\nRetrieved agronomic context:\n{rag_context}" if rag_context.strip() else ""
        return (
            f"Visual report:\n{visual_report}"
            f"{rag_block}\n\n"
            f"Farmer's question:\n{farmer_query}\n\n"
            f"Answer:"
        )
