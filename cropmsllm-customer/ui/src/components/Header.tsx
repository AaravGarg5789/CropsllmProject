import type { ApiState } from "../types";
import { Cpu, Layers, Target } from "lucide-react";

interface Props {
  state: ApiState | null;
}

const fmtPct = (x: number | null | undefined, digits = 2) =>
  x == null ? "-" : `${(x * 100).toFixed(digits)}%`;

export default function Header({ state }: Props) {
  return (
    <header className="border-b border-line bg-white">
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="flex items-center gap-3">
          <Layers className="h-6 w-6 text-emerald-600" />
          <h1 className="text-2xl font-semibold tracking-tight text-ink">
            AgriMTL
          </h1>
          <span className="text-sm text-muted">Agricultural Multi-Task Learning</span>
        </div>
        <p className="mt-2 max-w-3xl text-sm text-muted">
          Multi-task learning over multispectral UAV imagery: crop / weed segmentation,
          NDVI / NDRE regression, NDVI-binned stress severity, field zoning, and an
          LLM-grounded farmer advice layer.
        </p>

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <Stat
            icon={<Target className="h-4 w-4" />}
            label="3-class mIoU (test)"
            value={fmtPct(state?.headline_test.mean_iou_3class)}
            sub={state ? `${state.headline_test.n_tiles ?? "-"} tiles` : ""}
          />
          <Stat
            icon={<Target className="h-4 w-4" />}
            label="Pixel accuracy"
            value={fmtPct(state?.headline_test.pixel_accuracy)}
            sub="canonical test"
          />
          <Stat
            icon={<Cpu className="h-4 w-4" />}
            label="Encoder"
            value={state ? state.encoder.toUpperCase() : "-"}
            sub={state?.train_hw ? `train ${state.train_hw[0]}×${state.train_hw[1]}` : ""}
          />
        </div>

        <p className="mt-4 text-xs text-muted">
          MiT-B3 SegFormer encoder (45.2 M params) + DeepLabv3+ decoder + 4 heads
          (seg / NDVI / NDRE / canopy). Trained at 576×576, predicts at native 600×600
          via auto-resize. 4-flip TTA at inference. Self-supervised auxiliary loss
          with Kendall-Gal uncertainty weighting.
        </p>
      </div>
    </header>
  );
}

function Stat({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="card flex items-center gap-4 py-4">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
        {icon}
      </div>
      <div>
        <div className="h-eyebrow">{label}</div>
        <div className="mt-0.5 text-xl font-semibold tabular-nums tracking-tight">{value}</div>
        {sub ? <div className="text-xs text-muted">{sub}</div> : null}
      </div>
    </div>
  );
}
