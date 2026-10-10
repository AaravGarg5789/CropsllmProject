import { useEffect, useState } from "react";
import { api } from "./api";
import type { ApiState, PredictResponse, SamplesResponse } from "./types";
import Header from "./components/Header";
import InputPanel from "./components/InputPanel";
import VisualOutputs from "./components/VisualOutputs";
import TaskOutputs from "./components/TaskOutputs";
import LLMPanel from "./components/LLMPanel";
import PrescriptionPanel from "./components/PrescriptionPanel";
import MetricsPanel from "./components/MetricsPanel";
import ArchitecturePanel from "./components/ArchitecturePanel";

const DEFAULT_QUERY =
  "Where are the weeds in this maize field and what should I spray?";

export default function App() {
  const [state, setState] = useState<ApiState | null>(null);
  const [samples, setSamples] = useState<SamplesResponse | null>(null);
  const [tileId, setTileId] = useState("");
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [anomalyMode, setAnomalyMode] = useState("weed");
  const [ollamaModel, setOllamaModel] = useState("llama3.1:8b");

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [llmAnswer, setLlmAnswer] = useState<string | null>(null);
  const [llmBusy, setLlmBusy] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);

  useEffect(() => {
    api.state().then(setState).catch((e) => setError(String(e)));
    api.samples().then(setSamples).catch(() => null);
  }, []);

  const askLLM = async (visualReport: string, q: string, model: string) => {
    setLlmBusy(true);
    setLlmError(null);
    setLlmAnswer(null);
    try {
      const res = await api.llmAsk(visualReport, q, model);
      setLlmAnswer(res.answer);
    } catch (e) {
      setLlmError(e instanceof Error ? e.message : String(e));
    } finally {
      setLlmBusy(false);
    }
  };

  const run = async () => {
    if (!tileId) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.predict(tileId, anomalyMode);
      setResult(r);
      // Auto-ask the LLM with the current question + fresh visual_report.
      // The user can re-ask manually from the LLM panel without re-running predict.
      askLLM(r.visual_report, query, ollamaModel);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas/40">
      <Header state={state} />

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        {error ? (
          <div className="card border-red-200 bg-red-50 text-sm text-red-700">
            {error}
          </div>
        ) : null}

        <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
          <InputPanel
            samples={samples}
            tileId={tileId}
            setTileId={setTileId}
            query={query}
            setQuery={setQuery}
            anomalyMode={anomalyMode}
            setAnomalyMode={setAnomalyMode}
            ollamaModel={ollamaModel}
            setOllamaModel={setOllamaModel}
            onRun={run}
            loading={loading}
          />
          <VisualOutputs result={result} loading={loading} />
        </div>

        <TaskOutputs result={result} loading={loading} />

        <LLMPanel
          result={result}
          query={query}
          ollamaModel={ollamaModel}
          answer={llmAnswer}
          busy={llmBusy}
          error={llmError}
          onAsk={() => result && askLLM(result.visual_report, query, ollamaModel)}
        />

        <PrescriptionPanel result={result} loading={loading} />

        <MetricsPanel />
        <ArchitecturePanel state={state} />

        <footer className="pt-4 text-xs text-muted">
          AgriMTL · Vite + React + FastAPI ·{" "}
          <code>{state?.ckpt_name ?? "loading…"}</code>
        </footer>
      </main>
    </div>
  );
}
