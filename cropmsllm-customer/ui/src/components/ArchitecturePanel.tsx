import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { api } from "../api";
import type { ApiState, ArchitectureResponse } from "../types";

interface Props {
  state: ApiState | null;
}

export default function ArchitecturePanel({ state }: Props) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ArchitectureResponse | null>(null);

  useEffect(() => {
    if (!open || data) return;
    api.architecture().then(setData).catch(() => null);
  }, [open, data]);

  return (
    <section className="card">
      <button
        type="button"
        className="flex w-full items-center justify-between text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="h-section">Architecture · pipeline details</span>
        {open ? (
          <ChevronDown className="h-4 w-4 text-muted" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted" />
        )}
      </button>

      {open ? (
        <div className="mt-4 space-y-5 text-sm">
          {data ? (
            <>
              <div>
                <div className="h-eyebrow mb-1.5">Pipeline</div>
                <pre className="overflow-auto rounded-lg border border-line bg-canvas p-3 font-mono text-xs leading-relaxed text-ink">
                  {data.pipeline}
                </pre>
              </div>
              <div>
                <div className="h-eyebrow mb-1.5">Dataset (WeedsGalore)</div>
                <p className="text-muted">{data.dataset}</p>
              </div>
              {data.operating_range ? (
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-emerald-900">
                  <div className="h-eyebrow mb-1 text-emerald-800">Operating range</div>
                  <p>{data.operating_range}</p>
                </div>
              ) : null}
            </>
          ) : (
            <p className="text-muted">Loading…</p>
          )}
          {state ? (
            <div>
              <div className="h-eyebrow mb-1.5">Released checkpoint</div>
              <p className="text-muted">
                <code>runs/{state.ckpt_name}</code> - encoder{" "}
                <code>{state.encoder}</code>, train resolution{" "}
                {state.train_hw ? `${state.train_hw[0]}×${state.train_hw[1]}` : "-"},
                45.2 M params. Native {state.num_classes}-class output; UI headline
                collapses to 3-class (background / maize / merged-weed) per the
                WeedsGalore convention.
              </p>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
