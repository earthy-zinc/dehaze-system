import request from "@/utils/request";
import type {
  AiEvalAgentOverviewItem,
  AiEvalJudgeStatus,
  AiEvalReviewQueueResult,
  AiEvalReviewsQuery,
  AiEvalReviewSubmitForm,
  AiEvalReviewSubmitResult,
  AiEvalRunCompareResult,
  AiEvalTrendItem,
  AiEvalTrendsQuery,
} from "./model";

/** 管理端权限标识（后端拦截，无权限返回 403） */
const MANAGE_PERMISSION = "ai:agent:manage";

/**
 * AI 评测中心 API
 *
 * 内部 API（`/api/v1/ai/eval-center`），跨 Agent 聚合视角：各 Agent 最近得分与
 * 门禁状态、历史趋势、两次 run 对比、判分模型状态、人工复核队列与回填。
 *
 * 全部端点需 `ai:agent:manage` 权限，由后端拦截。
 * 单 Agent 的评测集/样本/执行记录见 `AiAgentAPI`，评测触发与发布门禁同在智能体管理。
 */
class AiEvalAPI {
  /** 评测总览（各 Agent 最近得分/门禁状态/退化标识） */
  static getOverview() {
    return request<AiEvalAgentOverviewItem[]>({
      url: "/api/v1/ai/eval-center/overview",
      method: "get",
    });
  }

  /** 评测历史趋势（按 Agent/时间范围过滤） */
  static getTrends(query?: AiEvalTrendsQuery) {
    return request<AiEvalTrendItem[]>({
      url: "/api/v1/ai/eval-center/trends",
      method: "get",
      params: query,
    });
  }

  /** 两次评测 run 得分对比（四维差异 + 样本级差异） */
  static compareRuns(runId: number, baseRunId: number) {
    return request<AiEvalRunCompareResult>({
      url: `/api/v1/ai/eval-center/runs/${runId}/compare`,
      method: "get",
      params: { baseRunId },
    });
  }

  /** 判分模型状态（一致性/漂移/门禁暂停提示） */
  static getJudgeStatus() {
    return request<AiEvalJudgeStatus>({
      url: "/api/v1/ai/eval-center/judge-status",
      method: "get",
    });
  }

  /** 人工复核队列（失败样本全量 + 通过样本按比例抽样） */
  static getReviews(query?: AiEvalReviewsQuery) {
    return request<AiEvalReviewQueueResult>({
      url: "/api/v1/ai/eval-center/reviews",
      method: "get",
      params: query,
    });
  }

  /** 复核结果回填（判定一致/不一致 + 备注） */
  static submitReview(id: number, data: AiEvalReviewSubmitForm) {
    return request<AiEvalReviewSubmitResult>({
      url: `/api/v1/ai/eval-center/reviews/${id}`,
      method: "post",
      data,
    });
  }
}

export default AiEvalAPI;
export { MANAGE_PERMISSION };
