import { AlertCircle, AlertTriangle, CheckCircle2 } from "lucide-react";
import type {
  PredictResponse,
  SeverityLabel,
  ZoneCell,
  ZoneClass,
} from "../types";
import { Swatch } from "./VisualOutputs";

interface Props {
  result: PredictResponse | null;
  loading: boolean;
}

const SEVERITY_BINS: { label: SeverityLabel; range: string; color: string }[] = [
  { label: "Severe",   range: "NDVI < 0.2",   color: "#c83220" },
  { label: "Moderate", range: "0.2 - 0.4",    color: "#f0b432" },
  { label: "Mild",     range: "≥ 0.4",        color: "#46aa46" },
];

const SEVERITY_KEY: Record<SeverityLabel, keyof PredictResponse["severity"]["breakdown_pct"]> = {
  Severe: "severe",
  Moderate: "moderate",
  Mild: "mild",
};

const SEVERITY_TONE: Record<SeverityLabel, { bg: string; ring: string; text: string }> = {
  Severe:   { bg: "bg-red-50",     ring: "ring-red-200",     text: "text-red-700" },
  Moderate: { bg: "bg-amber-50",   ring: "ring-amber-200",   text: "text-amber-700" },
  Mild:     { bg: "bg-emerald-50", ring: "ring-emerald-200", text: "text-emerald-700" },
};

const ZONE_TONE: Record<ZoneClass, { bg: string; ring: string; text: string; icon: React.ReactNode }> = {
  Safe:      { bg: "bg-emerald-50", ring: "ring-emerald-200", text: "text-emerald-700",
               icon: <CheckCircle2 className="h-4 w-4" /> },
  Attention: { bg: "bg-amber-50",   ring: "ring-amber-200",   text: "text-amber-700",
               icon: <AlertTriangle className="h-4 w-4" /> },
  Critical:  { bg: "bg-red-50",     ring: "ring-red-200",     text: "text-red-700",
               icon: <AlertCircle className="h-4 w-4" /> },
};

export default function TaskOutputs({ result, loading }: Props) {
  return (
    <section className="card">
      <h2 className="h-section">Stress severity · Field zoning · Actions</h2>
      <p className="mt-1 text-sm text-muted">
        Pipeline: per-pixel NDVI → 3-class severity → 3×3 zone classification →
        per-zone action recommendation.
      </p>

      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        <SeverityCard result={result} loading={loading} />
        <ZoningCard result={result} loading={loading} />
        <ActionsCard result={result} loading={loading} />
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <ImageBlock
          title="NDVI / Health intensity"
          url={result?.images.health}
          loading={loading}
        />
        <ImageBlock
          title="Stress type"
          url={result?.images.stress}
          loading={loading}
          legend={
            <>
              <Swatch color="#FFDC14" /> N-def <Swatch color="#3282FF" /> Water{" "}
              <Swatch color="#C832C8" /> General
            </>
          }
        />
      </div>

      {result ? (
        <div className="mt-4 rounded-lg border border-line bg-canvas/40 p-3 text-sm">
          <div className="h-eyebrow mb-1">Dominant stress type</div>
          <p className="text-muted">
            <strong className="text-ink">{result.tasks.stress_type}</strong>{" "}
            · heuristic: N-def = NDRE&lt;0.20 ∧ NDVI&gt;0.30; Water = NDVI&lt;0.30 ∧
            NIR&lt;median; General otherwise.
          </p>
        </div>
      ) : null}
    </section>
  );
}

function SeverityCard({ result, loading }: Props) {
  const sev = result?.severity;
  const tone = sev ? SEVERITY_TONE[sev.label] : null;
  return (
    <div className="rounded-lg border border-line bg-canvas/40 p-4">
      <div className="flex items-baseline justify-between">
        <div className="h-eyebrow">Stress severity (3-class)</div>
        {sev ? (
          <span className="text-xs text-muted">
            mean NDVI <span className="tabular-nums">{sev.mean_ndvi.toFixed(3)}</span>
          </span>
        ) : null}
      </div>

      {sev && tone ? (
        <div
          className={`mt-2 inline-flex items-center rounded-md px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${tone.bg} ${tone.ring} ${tone.text}`}
        >
          Dominant: {sev.label}
        </div>
      ) : null}

      <div className="mt-3 aspect-square overflow-hidden rounded-lg border border-line bg-white">
        {result?.images.severity ? (
          <img
            src={result.images.severity}
            alt="Per-pixel severity"
            className="h-full w-full object-cover"
          />
        ) : (
          <Placeholder loading={loading} />
        )}
      </div>

      <div className="mt-3 space-y-1.5">
        {SEVERITY_BINS.map((b) => {
          const pct = sev ? sev.breakdown_pct[SEVERITY_KEY[b.label]] : 0;
          return (
            <div key={b.label} className="flex items-center gap-2 text-xs">
              <span
                className="inline-block h-3 w-3 flex-shrink-0 rounded-sm"
                style={{ backgroundColor: b.color }}
              />
              <span className="w-16 font-medium text-ink">{b.label}</span>
              <span className="w-16 text-muted">{b.range}</span>
              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-canvas">
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.max(pct, 0).toFixed(1)}%`,
                    backgroundColor: b.color,
                  }}
                />
              </div>
              <span className="w-12 text-right tabular-nums text-muted">
                {sev ? pct.toFixed(1) : "-"}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ZoningCard({ result, loading }: Props) {
  const z = result?.zoning;
  return (
    <div className="rounded-lg border border-line bg-canvas/40 p-4">
      <div className="flex items-baseline justify-between">
        <div className="h-eyebrow">Field zoning (3×3 = 9 zones)</div>
        {z ? (
          <span className="text-xs text-muted">
            of {z.total_zones} zones
          </span>
        ) : null}
      </div>

      {z ? (
        <div className="mt-2 flex flex-wrap gap-1.5">
          <ZoneCount label="Critical" tone={ZONE_TONE.Critical} count={z.counts.critical} />
          <ZoneCount label="Attention" tone={ZONE_TONE.Attention} count={z.counts.attention} />
          <ZoneCount label="Safe" tone={ZONE_TONE.Safe} count={z.counts.safe} />
        </div>
      ) : null}

      <div className="mt-3 aspect-square overflow-hidden rounded-lg border border-line bg-white">
        {result?.images.zoning ? (
          <img
            src={result.images.zoning}
            alt="Zone A/B/C classification"
            className="h-full w-full object-cover"
          />
        ) : (
          <Placeholder loading={loading} />
        )}
      </div>

      <p className="mt-2 text-xs text-muted">
        Critical: ≥30% Severe pixels · Attention: ≥30% Moderate · Safe: otherwise.
      </p>
    </div>
  );
}

function ZoneCount({
  label,
  tone,
  count,
}: {
  label: ZoneClass;
  tone: typeof ZONE_TONE[ZoneClass];
  count: number;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${tone.bg} ${tone.ring} ${tone.text}`}
    >
      {tone.icon}
      {count} {label}
    </span>
  );
}

function ActionsCard({ result, loading }: Props) {
  const zones = result?.zoning?.zones ?? [];
  const actionable = zones.filter((z) => z.classification !== "Safe");
  return (
    <div className="rounded-lg border border-line bg-canvas/40 p-4">
      <div className="h-eyebrow">Recommended actions</div>
      {!result ? (
        <p className="mt-3 text-xs text-muted">
          {loading ? "Computing…" : "Run the pipeline to populate"}
        </p>
      ) : actionable.length === 0 ? (
        <div className="mt-3 flex items-start gap-2 rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800 ring-1 ring-inset ring-emerald-200">
          <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0" />
          <span>
            No actionable zones - all 9 zones classified as <strong>Safe</strong>.
          </span>
        </div>
      ) : (
        <ul className="mt-3 space-y-2 text-sm">
          {actionable.map((z) => (
            <ActionItem key={`${z.row}-${z.col}`} zone={z} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ActionItem({ zone }: { zone: ZoneCell }) {
  const tone = ZONE_TONE[zone.classification];
  return (
    <li className={`rounded-lg p-3 ring-1 ring-inset ${tone.bg} ${tone.ring}`}>
      <div className="flex items-center gap-2">
        <span className={tone.text}>{tone.icon}</span>
        <span className={`text-xs font-semibold uppercase tracking-wider ${tone.text}`}>
          {zone.classification}
        </span>
        <span className="ml-auto rounded-md bg-white/60 px-1.5 py-0.5 text-xs font-medium text-ink">
          {zone.cell_label}
        </span>
      </div>
      <p className="mt-1.5 text-xs text-ink">{zone.action}</p>
      <p className="mt-1 text-xs text-muted">
        Severe {zone.pct_severe.toFixed(0)}% · Moderate {zone.pct_moderate.toFixed(0)}%{" "}
        · Mild {zone.pct_mild.toFixed(0)}%
      </p>
    </li>
  );
}

function ImageBlock({
  title,
  url,
  loading,
  legend,
}: {
  title: string;
  url?: string;
  loading: boolean;
  legend?: React.ReactNode;
}) {
  return (
    <div>
      <div className="h-eyebrow mb-1.5">{title}</div>
      <div className="aspect-square overflow-hidden rounded-lg border border-line bg-canvas">
        {url ? (
          <img src={url} alt={title} className="h-full w-full object-cover" />
        ) : (
          <Placeholder loading={loading} />
        )}
      </div>
      {legend ? <div className="mt-2 text-xs text-muted">{legend}</div> : null}
    </div>
  );
}

function Placeholder({ loading }: { loading: boolean }) {
  return (
    <div className="flex h-full items-center justify-center text-xs text-muted">
      {loading ? "Computing…" : "Run the pipeline to populate"}
    </div>
  );
}
