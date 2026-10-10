import { Download, Map } from "lucide-react";
import type { PredictResponse } from "../types";

interface Props {
  result: PredictResponse | null;
  loading: boolean;
}

export default function PrescriptionPanel({ result, loading }: Props) {
  return (
    <section className="card">
      <div className="flex items-baseline justify-between">
        <h2 className="h-section">Prescription map (saved GeoTIFF)</h2>
        {result ? (
          <a
            href={result.geotiff_url}
            download
            className="btn-ghost"
            title="Download 2-band GeoTIFF (band 1 = seg classes, band 2 = anomaly mask)"
          >
            <Download className="h-4 w-4" /> GeoTIFF
          </a>
        ) : null}
      </div>
      <p className="mt-1 text-sm text-muted">
        Read back from the 2-band GeoTIFF actually written to disk: band 1 = seg
        classes (colorized), band 2 = anomaly mask (yellow tint blended). This is the
        georeferenced output a farmer / drone-spray system would consume.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-[2fr_1fr]">
        <div className="aspect-square overflow-hidden rounded-lg border border-line bg-canvas">
          {result?.images.prescription_preview ? (
            <img
              src={result.images.prescription_preview}
              alt="Prescription preview"
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted">
              {loading ? "Computing…" : "Run the pipeline to populate"}
            </div>
          )}
        </div>

        <div className="space-y-3 text-sm">
          <div className="rounded-lg border border-line bg-canvas/50 p-3">
            <div className="h-eyebrow flex items-center gap-1.5">
              <Map className="h-3.5 w-3.5" /> File
            </div>
            <p className="mt-1 break-all font-mono text-xs">
              {result?.geotiff_url ?? "-"}
            </p>
          </div>
          {result ? (
            <div className="rounded-lg border border-line bg-canvas/50 p-3 text-xs text-muted">
              Anomaly mode: <code>{result.anomaly_mode}</code>
              <br />
              Canopy estimate: {(result.canopy_pred * 100).toFixed(1)}%
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
