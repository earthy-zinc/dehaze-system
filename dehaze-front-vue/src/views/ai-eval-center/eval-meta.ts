import type { AiEvalConsistencyState, AiEvalGateStatus } from "dehaze-sdk-js";

/** 四维评分指标（键为后端 snake_case 指标名，dimensions/dimensionDiff 原样返回） */
export const EVAL_DIMENSIONS = [
  { key: "result_quality", label: "结果质量" },
  { key: "process_compliance", label: "过程合规" },
  { key: "safety_boundary", label: "安全边界" },
  { key: "efficiency", label: "效率" },
];

/** 门禁状态：passed 通过 / failed 未通过 / none 未评测 */
export const GATE_STATUS_META: Record<
  AiEvalGateStatus,
  { label: string; type: "success" | "danger" | "info" }
> = {
  passed: { label: "通过", type: "success" },
  failed: { label: "未通过", type: "danger" },
  none: { label: "未评测", type: "info" },
};

/** 评测执行状态：1 执行中 / 2 通过 / 3 失败 */
export const RUN_STATUS_META: Record<
  number,
  { label: string; type: "warning" | "success" | "danger" }
> = {
  1: { label: "执行中", type: "warning" },
  2: { label: "通过", type: "success" },
  3: { label: "失败", type: "danger" },
};

/** 触发方式 */
export const TRIGGER_TYPE_META: Record<string, string> = {
  manual: "手动触发",
  publish: "发布门禁",
};

/** 判分一致性状态 */
export const CONSISTENCY_STATE_META: Record<
  AiEvalConsistencyState,
  { label: string; type: "success" | "danger" | "info"; desc: string }
> = {
  normal: {
    label: "正常",
    type: "success",
    desc: "人工复核一致率达标，判分结果可信",
  },
  drifted: {
    label: "漂移",
    type: "danger",
    desc: "人工复核一致率低于阈值，判分结果可信度下降",
  },
  insufficient_data: {
    label: "样本不足",
    type: "info",
    desc: "尚无已复核样本，无法评估判分一致性",
  },
};

/** 样本风险等级 */
export const RISK_LEVEL_META: Record<
  string,
  { label: string; type: "info" | "warning" | "danger" }
> = {
  low: { label: "低", type: "info" },
  medium: { label: "中", type: "warning" },
  high: { label: "高", type: "danger" },
};

/** 评测评分聚合（score_summary 解包结果） */
export interface EvalScoreSummary {
  dimensions: Record<string, number>;
  sampleCount: number;
  passedCount: number;
  failedCount: number;
  /** 样本通过率（0-1） */
  passRate: number | null;
}

/** 样本执行明细（results 解包结果） */
export interface EvalSampleItem {
  sampleId: number;
  taskGoal: string;
  riskLevel: string;
  passed: boolean;
  error: string | null;
  scores: Record<string, number>;
  notes: Record<string, string>;
  metrics: {
    steps: number;
    latencyMs: number;
    inputTokens: number;
    outputTokens: number;
  };
  totalScore: number | null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asNumberMap(value: unknown): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [key, item] of Object.entries(asRecord(value))) {
    if (typeof item === "number") {
      result[key] = item;
    }
  }
  return result;
}

function asStringMap(value: unknown): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(asRecord(value))) {
    if (typeof item === "string") {
      result[key] = item;
    }
  }
  return result;
}

/** 四维得分均值（与后端口径一致：dimensions 内所有得分取均值） */
export function averageScore(dimensions?: Record<string, number> | null) {
  const values = Object.values(dimensions ?? {});
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function parseScoreSummary(raw: unknown): EvalScoreSummary {
  const summary = asRecord(raw);
  return {
    dimensions: asNumberMap(summary.dimensions),
    sampleCount: Number(summary.sample_count ?? 0),
    passedCount: Number(summary.passed_count ?? 0),
    failedCount: Number(summary.failed_count ?? 0),
    passRate: summary.pass_rate == null ? null : Number(summary.pass_rate),
  };
}

export function parseSamples(raw: unknown): EvalSampleItem[] {
  const list = Array.isArray(raw) ? raw : [];
  return list.flatMap((item) => {
    const source = asRecord(item);
    const sampleId = source.sample_id;
    if (typeof sampleId !== "number") return [];
    const metrics = asRecord(source.metrics);
    const scores = asNumberMap(source.scores);
    return [
      {
        sampleId,
        taskGoal: String(source.task_goal ?? ""),
        riskLevel: String(source.risk_level ?? "low"),
        passed: Boolean(source.passed),
        error: source.error == null ? null : String(source.error),
        scores,
        notes: asStringMap(source.notes),
        metrics: {
          steps: Number(metrics.steps ?? 0),
          latencyMs: Number(metrics.latency_ms ?? 0),
          inputTokens: Number(metrics.input_tokens ?? 0),
          outputTokens: Number(metrics.output_tokens ?? 0),
        },
        totalScore: averageScore(scores),
      },
    ];
  });
}

export function formatScore(score?: number | null) {
  return score == null ? "-" : score.toFixed(2);
}

/** 通过率（0-1）转百分比展示 */
export function formatRate(rate?: number | null) {
  return rate == null ? "-" : `${(rate * 100).toFixed(1)}%`;
}

export function formatDuration(ms?: number | null) {
  if (ms == null) return "-";
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/** 时间展示：后端返回 YYYY-MM-DD HH:mm:ss，截断到分钟 */
export function formatTime(time?: string | null) {
  if (!time) return "-";
  return time.length > 16 ? time.slice(0, 16) : time;
}
