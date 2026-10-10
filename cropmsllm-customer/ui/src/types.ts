export type SeverityLabel = "Severe" | "Moderate" | "Mild";
export type ZoneClass = "Safe" | "Attention" | "Critical";

export interface SampleEntry {
  tile_id: string;
  ndvi: number;
}

export interface SamplesResponse {
  tiles: SampleEntry[];
  total: number;
}

export interface ApiState {
  ckpt_name: string;
  encoder: string;
  num_classes: number;
  train_hw: [number, number] | null;
  headline_test: {
    n_tiles: number | null;
    mean_iou_3class: number | null;
    pixel_accuracy: number | null;
    ndvi_mae: number | null;
    ndre_mae: number | null;
    task_ndvi_rmse: number | null;
    task_stress_accuracy: number | null;
    task_stress_macro_f1: number | null;
    task_zone_mean_iou: number | null;
  };
}

export interface SeveritySummary {
  label: SeverityLabel;
  mean_ndvi: number;
  breakdown_pct: { severe: number; moderate: number; mild: number };
  actual_label: SeverityLabel;
  actual_mean_ndvi: number;
}

export interface ZoneCell {
  row: number;
  col: number;
  cell_label: string;
  classification: ZoneClass;
  pct_severe: number;
  pct_moderate: number;
  pct_mild: number;
  action: string;
}

export interface ZoningSummary {
  grid_n: number;
  total_zones: number;
  counts: { safe: number; attention: number; critical: number };
  zones: ZoneCell[];
}

export interface PredictResponse {
  request_id: string;
  tile_id: string;
  elapsed_s: number;
  anomaly_mode: string;
  num_classes: number;
  images: {
    rgb: string;
    seg: string;
    anomaly: string;
    severity: string;
    zoning: string;
    health: string;
    stress: string;
    prescription_preview: string;
  };
  severity: SeveritySummary;
  zoning: ZoningSummary;
  tasks: {
    stress_type: string;
    spatial_text: string;
  };
  canopy_pred: number;
  visual_report: string;
  geotiff_url: string;
}

export interface SplitMetrics {
  n_tiles: number;
  pixel_accuracy: number;
  mean_iou_3class: number;
  mean_iou_6class?: number;
  ndvi_mae: number;
  ndre_mae: number;
  canopy_mae: number;
  task_health_f1?: number;
  task_ndvi_rmse?: number;
  task_stress_accuracy?: number;
  task_stress_macro_f1?: number;
  task_zone_mean_iou?: number;
}

export interface SeverityMetrics {
  split: string;
  n_tiles: number;
  bins_3class: { severe: string; moderate: string; mild: string };
  per_pixel_accuracy: {
    mean_3class: number;
    min_3class: number;
    median_3class: number;
    max_3class: number;
    mean_4class: number;
  };
  per_class: Record<string, { precision: number; recall: number; f1: number }>;
  macro_f1_3class: number;
  pixel_distribution_pct: { severe: number; moderate: number; mild: number };
}

export interface BenchmarkPaper {
  title: string;
  authors?: string;
  year?: number;
  venue?: string;
  architecture: string;
  params_m?: number | null;
  input_bands?: string;
  input_res?: string;
  dataset?: string;
  primary_metric: string;
  multitask?: string;
  url?: string;
}

export interface BenchmarksBlock {
  context?: string;
  ours: BenchmarkPaper;
  papers: BenchmarkPaper[];
  positioning?: string[];
}

export interface MetricsResponse {
  num_classes: number;
  encoder: string;
  train_hw: [number, number];
  splits: Record<string, SplitMetrics>;
  severity?: SeverityMetrics;
  benchmarks?: BenchmarksBlock;
}

export interface ArchitectureResponse {
  pipeline: string;
  dataset: string;
  operating_range?: string;
}

export interface LLMResponse {
  model: string;
  answer: string;
}
