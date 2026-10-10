import { Play } from "lucide-react";
import type { SamplesResponse } from "../types";

interface Props {
  samples: SamplesResponse | null;
  tileId: string;
  setTileId: (s: string) => void;
  query: string;
  setQuery: (s: string) => void;
  anomalyMode: string;
  setAnomalyMode: (s: string) => void;
  ollamaModel: string;
  setOllamaModel: (s: string) => void;
  onRun: () => void;
  loading: boolean;
}

const EXAMPLE_QUERIES = [
  "Where are the weeds in this maize field and what should I spray?",
  "Is this maize field healthy enough to skip herbicide application this week?",
  "Give me a 2-sentence summary I can read out to a farmer.",
  "Which area of this field would you mechanically weed first?",
  "Explain whether the weed pressure justifies a full-field spray vs. a targeted one.",
];

const ANOMALY_MODES: { value: string; label: string }[] = [
  { value: "weed", label: "weed (ExG + NGRDI)" },
  { value: "stress", label: "stress (NDVI + NDRE)" },
  { value: "uncertainty", label: "uncertainty (entropy)" },
  { value: "union", label: "union (OR)" },
];

const OLLAMA_MODELS = ["llama3.1:8b", "llama3.2:3b", "qwen3:8b", "mistral:7b"];

export default function InputPanel({
  samples,
  tileId,
  setTileId,
  query,
  setQuery,
  anomalyMode,
  setAnomalyMode,
  ollamaModel,
  setOllamaModel,
  onRun,
  loading,
}: Props) {
  const total = samples?.total ?? 0;
  return (
    <section className="card">
      <h2 className="h-section">Input</h2>
      <p className="mt-1 text-sm text-muted">
        Canonical test split - {total} tiles, all in the model's valid operating
        range (NDVI &lt; 0.6).
      </p>

      <div className="mt-5 space-y-4">
        <Field label={`Test tile (${total})`}>
          <select
            className="input-base"
            value={tileId}
            onChange={(e) => setTileId(e.target.value)}
          >
            <option value="">(select a tile)</option>
            {samples?.tiles.map((t) => (
              <option key={t.tile_id} value={t.tile_id}>
                {t.tile_id} · NDVI {t.ndvi >= 0 ? "+" : ""}
                {t.ndvi.toFixed(2)}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Farmer question">
          <textarea
            className="input-base resize-none"
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                type="button"
                className="rounded-full border border-line bg-canvas px-2.5 py-1 text-xs text-muted hover:bg-emerald-50 hover:text-emerald-700"
                onClick={() => setQuery(q)}
              >
                {q.slice(0, 40)}…
              </button>
            ))}
          </div>
        </Field>

        <Field label="Anomaly mode">
          <div className="grid grid-cols-2 gap-2">
            {ANOMALY_MODES.map((m) => (
              <label
                key={m.value}
                className={`cursor-pointer rounded-lg border px-3 py-2 text-xs ${
                  anomalyMode === m.value
                    ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                    : "border-line bg-white text-muted hover:bg-canvas"
                }`}
              >
                <input
                  type="radio"
                  name="anomaly"
                  className="sr-only"
                  checked={anomalyMode === m.value}
                  onChange={() => setAnomalyMode(m.value)}
                />
                {m.label}
              </label>
            ))}
          </div>
        </Field>

        <Field label="Ollama LLM (text-only Q&A)">
          <select
            className="input-base"
            value={ollamaModel}
            onChange={(e) => setOllamaModel(e.target.value)}
          >
            {OLLAMA_MODELS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>

        <button
          type="button"
          onClick={onRun}
          disabled={loading || !tileId}
          className="btn-primary w-full py-2.5"
        >
          <Play className="h-4 w-4" />
          {loading ? "Running pipeline…" : "Run pipeline"}
        </button>
      </div>
    </section>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="h-eyebrow mb-1.5">{label}</div>
      {children}
    </div>
  );
}
