import type { PredictResponse } from "../types";

interface Props {
  result: PredictResponse | null;
  loading: boolean;
}

export default function VisualOutputs({ result, loading }: Props) {
  return (
    <section className="card">
      <div className="flex items-baseline justify-between">
        <h2 className="h-section">Visual outputs</h2>
        {result ? (
          <span className="text-xs text-muted">
            {result.elapsed_s}s · tile <code>{result.tile_id}</code>
          </span>
        ) : null}
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <ImageCard title="RGB composite" url={result?.images.rgb} loading={loading} />
        <ImageCard
          title="Seg mask (3-class)"
          url={result?.images.seg}
          loading={loading}
        />
        <ImageCard
          title="Anomaly overlay (yellow flag)"
          url={result?.images.anomaly}
          loading={loading}
        />
      </div>

      <div className="mt-4 text-xs text-muted">
        <strong>Seg legend:</strong>{" "}
        <Swatch color="#323232" /> background <Swatch color="#1ec83c" /> maize{" "}
        <Swatch color="#dc2828" /> weed (any species)
      </div>
    </section>
  );
}

function ImageCard({
  title,
  url,
  loading,
}: {
  title: string;
  url?: string;
  loading: boolean;
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

export function Swatch({ color }: { color: string }) {
  return <span className="legend-swatch" style={{ backgroundColor: color }} />;
}
