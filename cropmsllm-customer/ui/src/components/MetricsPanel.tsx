import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { api } from "../api";
import type { BenchmarkPaper, MetricsResponse, SplitMetrics } from "../types";

const SPLIT_ORDER: (keyof MetricsResponse["splits"])[] = [
  "train",
  "train_full",
  "val",
  "test",
];

const SPLIT_LABELS: Record<string, string> = {
  train: "train (canonical 104)",
  train_full: "train_full (644)",
  val: "val (26)",
  test: "test (26 - headline)",
};

const fmtPct = (x: number | null | undefined, digits = 2) =>
  x == null ? "-" : `${(x * 100).toFixed(digits)}%`;
const fmtNum = (x: number | null | undefined, digits = 4) =>
  x == null ? "-" : x.toFixed(digits);

export default function MetricsPanel() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data) return;
    api
      .metrics()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open, data]);

  return (
    <section className="card">
      <button
        type="button"
        className="flex w-full items-center justify-between text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="h-section">
          Full accuracy report (segmentation + per-task + aux heads)
        </span>
        {open ? (
          <ChevronDown className="h-4 w-4 text-muted" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted" />
        )}
      </button>

      {open ? (
        <div className="mt-4 space-y-6">
          {error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : !data ? (
            <p className="text-sm text-muted">Loading…</p>
          ) : (
            <>
              <SegBlock data={data} />
              {data.severity ? <SeverityBlock data={data.severity} /> : null}
              <TaskBlock data={data} />
              <AuxBlock data={data} />
              {data.benchmarks ? <BenchmarksBlockView data={data.benchmarks} /> : null}
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}

function SegBlock({ data }: { data: MetricsResponse }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">Segmentation (primary, supervised)</h3>
      <p className="mt-0.5 text-xs text-muted">
        {data.encoder.toUpperCase()} encoder · trained at{" "}
        {data.train_hw[0]}×{data.train_hw[1]} · 4-flip TTA at eval.
      </p>
      <table className="mt-2 w-full text-sm">
        <thead className="border-b border-line text-xs uppercase tracking-wider text-muted">
          <tr>
            <Th>Split</Th>
            <Th right>Tiles</Th>
            <Th right>Pixel acc</Th>
            <Th right>3-cls mIoU</Th>
          </tr>
        </thead>
        <tbody>
          {SPLIT_ORDER.filter((s) => data.splits[s]).map((s) => {
            const m = data.splits[s] as SplitMetrics;
            return (
              <tr key={s} className="border-b border-line/60">
                <Td>{SPLIT_LABELS[s] ?? s}</Td>
                <Td right>{m.n_tiles}</Td>
                <Td right>{fmtPct(m.pixel_accuracy)}</Td>
                <Td right strong>
                  {fmtPct(m.mean_iou_3class)}
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TaskBlock({ data }: { data: MetricsResponse }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">
        Per-task metrics (deployed pipeline)
      </h3>
      <p className="mt-0.5 text-xs text-muted">
        Stress severity (3-class, NDVI bins) is the primary precision-ag task -
        see the dedicated block above. Stress type is the supplementary
        N-def / Water / General heuristic. Zone A/B/C is deterministic from severity
        (no separate metric).
      </p>
      <table className="mt-2 w-full text-sm">
        <thead className="border-b border-line text-xs uppercase tracking-wider text-muted">
          <tr>
            <Th>Split</Th>
            <Th right>NDVI RMSE</Th>
            <Th right>Stress type Acc / F1</Th>
          </tr>
        </thead>
        <tbody>
          {SPLIT_ORDER.filter((s) => data.splits[s]).map((s) => {
            const m = data.splits[s] as SplitMetrics;
            return (
              <tr key={s} className="border-b border-line/60">
                <Td>{s}</Td>
                <Td right>{fmtNum(m.task_ndvi_rmse)}</Td>
                <Td right>
                  {fmtPct(m.task_stress_accuracy)} / {fmtPct(m.task_stress_macro_f1)}
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AuxBlock({ data }: { data: MetricsResponse }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">
        Auxiliary regression heads (self-supervised)
      </h3>
      <table className="mt-2 w-full text-sm">
        <thead className="border-b border-line text-xs uppercase tracking-wider text-muted">
          <tr>
            <Th>Split</Th>
            <Th right>NDVI MAE</Th>
            <Th right>NDRE MAE</Th>
            <Th right>Canopy MAE</Th>
          </tr>
        </thead>
        <tbody>
          {SPLIT_ORDER.filter((s) => data.splits[s]).map((s) => {
            const m = data.splits[s] as SplitMetrics;
            return (
              <tr key={s} className="border-b border-line/60">
                <Td>{s}</Td>
                <Td right>{fmtNum(m.ndvi_mae, 3)}</Td>
                <Td right>{fmtNum(m.ndre_mae, 3)}</Td>
                <Td right>{fmtNum(m.canopy_mae, 3)}</Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SeverityBlock({ data }: { data: NonNullable<MetricsResponse["severity"]> }) {
  const ppa = data.per_pixel_accuracy;
  const pc = data.per_class;
  return (
    <div>
      <h3 className="text-sm font-semibold">
        Stress severity (3-class NDVI bins, derived from NDVI head)
      </h3>
      <p className="mt-0.5 text-xs text-muted">
        Severe (NDVI &lt; 0.2) · Moderate (0.2 - 0.4) · Mild (≥ 0.4). Computed on
        canonical {data.split} split, {data.n_tiles} tiles.
      </p>
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-line bg-canvas/40 p-3 text-sm">
          <div className="h-eyebrow mb-1.5">Per-pixel accuracy</div>
          <p className="text-2xl font-semibold tabular-nums">
            {fmtPct(ppa.mean_3class)}
          </p>
          <p className="text-xs text-muted">
            mean over {data.n_tiles} tiles · per-tile [{fmtPct(ppa.min_3class, 1)} /{" "}
            {fmtPct(ppa.median_3class, 1)} / {fmtPct(ppa.max_3class, 1)}]
          </p>
          <p className="mt-1.5 text-xs text-muted">
            Pixel distribution: Severe {data.pixel_distribution_pct.severe.toFixed(1)}%
            · Moderate {data.pixel_distribution_pct.moderate.toFixed(1)}%
            · Mild {data.pixel_distribution_pct.mild.toFixed(1)}%
          </p>
        </div>
        <div className="rounded-lg border border-line bg-canvas/40 p-3 text-sm">
          <div className="h-eyebrow mb-1.5">Per-class precision / recall / F1</div>
          <table className="w-full text-xs">
            <thead className="text-muted">
              <tr>
                <th className="text-left font-medium">Class</th>
                <th className="text-right font-medium">P</th>
                <th className="text-right font-medium">R</th>
                <th className="text-right font-medium">F1</th>
              </tr>
            </thead>
            <tbody>
              {(["Severe", "Moderate", "Mild"] as const).map((k) =>
                pc[k] ? (
                  <tr key={k} className="border-t border-line/60">
                    <td className="py-1">{k}</td>
                    <td className="text-right tabular-nums">
                      {fmtPct(pc[k].precision)}
                    </td>
                    <td className="text-right tabular-nums">
                      {fmtPct(pc[k].recall)}
                    </td>
                    <td className="text-right font-semibold tabular-nums">
                      {fmtPct(pc[k].f1)}
                    </td>
                  </tr>
                ) : null,
              )}
              <tr className="border-t border-line/60">
                <td className="py-1 font-medium">Macro-F1</td>
                <td colSpan={2}></td>
                <td className="text-right font-semibold tabular-nums">
                  {fmtPct(data.macro_f1_3class)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function BenchmarksBlockView({ data }: { data: NonNullable<MetricsResponse["benchmarks"]> }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">Benchmark - recent literature</h3>
      {data.context ? (
        <p className="mt-0.5 text-xs text-muted">{data.context}</p>
      ) : null}

      <table className="mt-3 w-full text-xs">
        <thead className="border-b border-line text-muted">
          <tr>
            <Th>Paper</Th>
            <Th>Year</Th>
            <Th>Architecture</Th>
            <Th>Bands</Th>
            <Th>Dataset</Th>
            <Th right>Headline metric</Th>
            <Th>Multi-task</Th>
          </tr>
        </thead>
        <tbody>
          <BenchmarkRow paper={data.ours} highlight />
          {data.papers.map((p, i) => (
            <BenchmarkRow paper={p} key={i} />
          ))}
        </tbody>
      </table>

      {data.positioning?.length ? (
        <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50/60 p-3 text-xs text-emerald-900">
          <div className="h-eyebrow mb-1 text-emerald-800">
            Positioning vs. literature
          </div>
          <ul className="list-disc space-y-0.5 pl-4">
            {data.positioning.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function BenchmarkRow({
  paper,
  highlight,
}: {
  paper: BenchmarkPaper;
  highlight?: boolean;
}) {
  const cls = highlight ? "bg-emerald-50/40 font-medium" : "";
  return (
    <tr className={`border-b border-line/60 ${cls}`}>
      <td className="px-2 py-2">
        {paper.url ? (
          <a
            href={paper.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-emerald-700 hover:underline"
          >
            {shorten(paper.title, 60)}
            <ExternalLink className="h-3 w-3" />
          </a>
        ) : (
          shorten(paper.title, 60)
        )}
        {paper.authors ? (
          <div className="text-muted">{paper.authors}</div>
        ) : null}
      </td>
      <td className="px-2 py-2 align-top">
        {paper.year ?? "-"}
        {paper.venue ? <div className="text-muted">{paper.venue}</div> : null}
      </td>
      <td className="px-2 py-2 align-top">
        {shorten(paper.architecture, 70)}
        {paper.params_m != null ? (
          <div className="text-muted">{paper.params_m} M params</div>
        ) : null}
      </td>
      <td className="px-2 py-2 align-top">{paper.input_bands ?? "-"}</td>
      <td className="px-2 py-2 align-top">{paper.dataset ?? "-"}</td>
      <td className="px-2 py-2 align-top text-right">{paper.primary_metric}</td>
      <td className="px-2 py-2 align-top">{paper.multitask ?? "-"}</td>
    </tr>
  );
}

function shorten(s: string, n: number) {
  return s.length <= n ? s : s.slice(0, n - 1) + "…";
}

function Th({
  children,
  right,
}: {
  children: React.ReactNode;
  right?: boolean;
}) {
  return (
    <th className={`px-3 py-1.5 ${right ? "text-right" : "text-left"} font-medium`}>
      {children}
    </th>
  );
}

function Td({
  children,
  right,
  strong,
}: {
  children: React.ReactNode;
  right?: boolean;
  strong?: boolean;
}) {
  return (
    <td
      className={`px-3 py-2 ${right ? "text-right tabular-nums" : ""} ${
        strong ? "font-semibold text-ink" : ""
      }`}
    >
      {children}
    </td>
  );
}
