import { PageResult } from "@/types";
import request from "@/utils/request";
import type {
  AiObservabilityCostsQuery,
  AiObservabilityCostsResult,
  AiObservabilitySummary,
  AiObservabilityTraceDetail,
  AiObservabilityTraceItem,
  AiObservabilityTraceQuery,
  AiObservabilityTrendItem,
  AiObservabilityTrendsQuery,
} from "./model";

/** 管理端审计权限标识（后端拦截，无权限返回 403） */
const AUDIT_PERMISSION = "ai:conversation:audit";

/**
 * AI 可观测性 API
 *
 * 内部 API（`/api/v1/ai/observability`），覆盖异常总览、过程链检索/详情/导出、
 * 资源消耗聚合与性能趋势，供管理端「AI 可观测中心」消费。
 *
 * 除过程链详情（登录用户可查，普通用户仅能查自己会话的过程链，跨会话返回 404）
 * 外，其余端点均需 `ai:conversation:audit` 权限，由后端拦截。
 */
class AiObservabilityAPI {
  /** 异常总览统计（失败/中断/超时/配额拒绝/高风险调用计数） */
  static getSummary() {
    return request<AiObservabilitySummary>({
      url: "/api/v1/ai/observability/summary",
      method: "get",
    });
  }

  /** 过程链检索（分页，支持会话/用户/状态/智能体/模型/时间筛选） */
  static getTraces(query?: AiObservabilityTraceQuery) {
    return request<PageResult<AiObservabilityTraceItem[]>>({
      url: "/api/v1/ai/observability/traces",
      method: "get",
      params: query,
    });
  }

  /** 过程链详情（trace 汇总 + 上下文快照 + LLM 调用回放） */
  static getTraceDetail(traceId: string) {
    return request<AiObservabilityTraceDetail>({
      url: `/api/v1/ai/observability/traces/${traceId}`,
      method: "get",
    });
  }

  /** 过程链导出（按检索条件全量导出 CSV，UTF-8 BOM），返回 Blob 下载 */
  static exportTraces(query?: AiObservabilityTraceQuery) {
    return request<Blob>({
      url: "/api/v1/ai/observability/traces/export",
      method: "get",
      params: query,
      responseType: "blob",
    });
  }

  /** 资源消耗聚合（按 model/agent/user 维度分页 + 按日 Token 趋势） */
  static getCosts(query?: AiObservabilityCostsQuery) {
    return request<AiObservabilityCostsResult>({
      url: "/api/v1/ai/observability/costs",
      method: "get",
      params: query,
    });
  }

  /** 性能趋势（按 model/agent 维度 + 日期聚合调用量/成功率/平均延迟） */
  static getTrends(query?: AiObservabilityTrendsQuery) {
    return request<AiObservabilityTrendItem[]>({
      url: "/api/v1/ai/observability/trends",
      method: "get",
      params: query,
    });
  }
}

export default AiObservabilityAPI;
export { AUDIT_PERMISSION };
