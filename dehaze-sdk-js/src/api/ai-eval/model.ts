// ==================== 枚举类型 ====================

/** 评测门禁状态 */
export type AiEvalGateStatus = "passed" | "failed" | "none";

/** 评测执行状态：2-通过，3-失败 */
export type AiEvalRunStatus = 2 | 3;

/** 评测触发方式 */
export type AiEvalTriggerType = "manual" | "publish";

/** 人工复核状态：1-待复核，2-已复核 */
export type AiEvalReviewStatus = 1 | 2;

/** 判分一致性状态 */
export type AiEvalConsistencyState = "normal" | "drifted" | "insufficient_data";

// ==================== 评测总览 ====================

/** 各 Agent 评测总览项 */
export interface AiEvalAgentOverviewItem {
  agentId: number;
  agentCode: string;
  agentName: string;
  /** 最近一次已完成评测记录 ID（未评测为 undefined） */
  runId?: number;
  runTime?: string;
  triggerType?: AiEvalTriggerType;
  gateStatus: AiEvalGateStatus;
  /** 四维总分（均值，0-100） */
  totalScore?: number;
  /** 四维得分（键为 snake_case 指标名，后端 dict 不做 camelCase 转换） */
  dimensions?: Record<string, number>;
  /** 相对上次评测是否退化（超过 ai_eval.regression_threshold） */
  degraded: boolean;
  /** 最近评测是否存在高风险样本失败 */
  highRiskFailed: boolean;
}

// ==================== 历史趋势 ====================

/** 评测历史趋势项 */
export interface AiEvalTrendItem {
  runId: number;
  agentId: number;
  agentName: string;
  triggerType: AiEvalTriggerType;
  status: AiEvalRunStatus;
  totalScore?: number;
  dimensions?: Record<string, number>;
  createTime?: string;
}

/** 评测历史趋势查询参数 */
export interface AiEvalTrendsQuery {
  agentId?: number;
  startTime?: string;
  endTime?: string;
  /** 返回条数，1-500，默认 100 */
  limit?: number;
}

// ==================== run 对比 ====================

/** 单次评测的得分快照 */
export interface AiEvalRunScoreSnapshot {
  runId: number;
  totalScore?: number;
  dimensions?: Record<string, number>;
  sampleCount: number;
  /** 样本通过率（0-1） */
  passRate?: number;
  createTime?: string;
}

/** 样本级差异项 */
export interface AiEvalSampleDiffItem {
  sampleId: number;
  taskGoal: string;
  currentPassed?: boolean;
  basePassed?: boolean;
  currentScore?: number;
  baseScore?: number;
  /** 总分变化（本次-基准） */
  scoreDelta?: number;
}

/** 样本级差异 */
export interface AiEvalSampleDiff {
  /** 仅本次评测包含的样本 */
  added: AiEvalSampleDiffItem[];
  /** 仅基准评测包含的样本 */
  removed: AiEvalSampleDiffItem[];
  /** 两次均包含但得分/通过状态有差异的样本 */
  changed: AiEvalSampleDiffItem[];
  unchangedCount: number;
}

/** 两次评测 run 对比结果 */
export interface AiEvalRunCompareResult {
  runId: number;
  baseRunId: number;
  agentId: number;
  current: AiEvalRunScoreSnapshot;
  base: AiEvalRunScoreSnapshot;
  /** 四维得分差（本次-基准），键为 snake_case 指标名 */
  dimensionDiff: Record<string, number>;
  sampleDiff: AiEvalSampleDiff;
}

// ==================== 判分模型状态 ====================

/** 人工复核统计 */
export interface AiEvalReviewStats {
  total: number;
  pending: number;
  reviewed: number;
  agreeCount: number;
  disagreeCount: number;
  /** 判分一致率（百分比 0-100，已复核口径） */
  agreementRate: number;
}

/** 判分模型状态（一致性/漂移/门禁暂停提示） */
export interface AiEvalJudgeStatus {
  consistencyState: AiEvalConsistencyState;
  /** 漂移门禁暂停提示（一致率低于阈值时为 true，提示暂停依赖判分的门禁判定） */
  driftPaused: boolean;
  /** 一致性阈值（百分比，sys_dict ai_eval） */
  consistencyThreshold: number;
  reviewStats: AiEvalReviewStats;
}

// ==================== 人工复核 ====================

/** 人工复核项 */
export interface AiEvalReviewItem {
  id: number;
  runId: number;
  sampleId: number;
  agentId: number;
  agentName?: string;
  /** 判分模型判定（true-通过，false-失败） */
  judgePassed: boolean;
  riskLevel: string;
  status: AiEvalReviewStatus;
  /** 人工判定（true-与判分一致，false-不一致，未复核为 undefined） */
  agree?: boolean;
  remark?: string;
  createTime?: string;
}

/** 人工复核队列 */
export interface AiEvalReviewQueueResult {
  /** 复核项列表（待复核优先） */
  items: AiEvalReviewItem[];
  pending: number;
  reviewed: number;
}

/** 人工复核队列查询参数 */
export interface AiEvalReviewsQuery {
  status?: AiEvalReviewStatus;
}

/** 人工复核回填表单 */
export interface AiEvalReviewSubmitForm {
  /** 人工判定（true-与判分一致，false-不一致） */
  agree: boolean;
  remark?: string;
}

/**
 * 人工复核回填结果。
 *
 * 后端 `POST /reviews/{id}` 声明为 `Result[dict]`，不经 OrmResult 的 camelCase
 * 别名转换，字段按服务层写入的 snake_case 原样返回（id 为唯一非下划线字段）。
 */
export interface AiEvalReviewSubmitResult {
  id: number;
  run_id: number;
  sample_id: number;
  agent_id: number;
  judge_passed: boolean;
  risk_level: string;
  status: AiEvalReviewStatus;
  agree: boolean;
  remark?: string;
}
