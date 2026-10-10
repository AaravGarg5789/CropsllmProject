import { Bot, FileText, Loader2, RefreshCw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { PredictResponse } from "../types";

interface Props {
  result: PredictResponse | null;
  query: string;
  ollamaModel: string;
  answer: string | null;
  busy: boolean;
  error: string | null;
  onAsk: () => void;
}

export default function LLMPanel({
  result,
  query,
  ollamaModel,
  answer,
  busy,
  error,
  onAsk,
}: Props) {
  return (
    <section className="card">
      <h2 className="h-section">LLM-grounded farmer answer</h2>
      <p className="mt-1 text-sm text-muted">
        Auto-fired after each pipeline run with the current question + the latest
        visual report. Re-ask anytime to regenerate (e.g. after editing the question
        or switching the Ollama model).
      </p>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <div>
          <div className="h-eyebrow mb-1.5 flex items-center gap-1.5">
            <FileText className="h-3.5 w-3.5" /> Visual report (sent to LLM)
          </div>
          {result ? (
            <pre className="max-h-72 overflow-auto rounded-lg border border-line bg-canvas p-3 font-mono text-xs text-ink">
              {result.visual_report}
            </pre>
          ) : (
            <p className="text-xs text-muted">Run the pipeline first.</p>
          )}
          {result ? (
            <p className="mt-2 text-xs text-muted">
              Question: <em className="text-ink">{query}</em>
            </p>
          ) : null}
        </div>

        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <div className="h-eyebrow flex items-center gap-1.5">
              <Bot className="h-3.5 w-3.5" /> Answer ({ollamaModel})
            </div>
            <button
              type="button"
              className="btn-ghost"
              disabled={!result || busy}
              onClick={onAsk}
              title="Re-run the LLM with the current question / model"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
              {busy ? "Asking…" : "Re-ask"}
            </button>
          </div>
          <div className="min-h-32 rounded-lg border border-line bg-canvas p-3 text-sm">
            {answer ? (
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{answer}</ReactMarkdown>
              </div>
            ) : busy ? (
              <p className="flex items-center gap-2 text-muted">
                <Loader2 className="h-4 w-4 animate-spin" /> Asking Ollama…
              </p>
            ) : error ? (
              <p className="text-red-600">{error}</p>
            ) : !result ? (
              <p className="text-muted">Run the pipeline to get an answer.</p>
            ) : (
              <p className="text-muted">No answer yet.</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
