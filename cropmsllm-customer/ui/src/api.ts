import type {
  ApiState,
  ArchitectureResponse,
  LLMResponse,
  MetricsResponse,
  PredictResponse,
  SamplesResponse,
} from "./types";

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  state: () => jsonFetch<ApiState>("/api/state"),
  samples: () => jsonFetch<SamplesResponse>("/api/samples"),
  metrics: () => jsonFetch<MetricsResponse>("/api/metrics"),
  architecture: () => jsonFetch<ArchitectureResponse>("/api/architecture"),
  predict: (tile_id: string, anomaly_mode: string) =>
    jsonFetch<PredictResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ tile_id, anomaly_mode }),
    }),
  llmAsk: (visual_report: string, question: string, model: string) =>
    jsonFetch<LLMResponse>("/api/llm/ask", {
      method: "POST",
      body: JSON.stringify({ visual_report, question, model }),
    }),
};
