import type { AiObservabilityStatus } from "dehaze-sdk-js";

/** 过程链执行状态展示（表格与详情抽屉共用） */
export const TRACE_STATUS_META: Record<
  AiObservabilityStatus,
  { label: string; tag: "success" | "danger" | "warning" | "primary" }
> = {
  1: { label: "成功", tag: "success" },
  2: { label: "失败", tag: "danger" },
  3: { label: "中断", tag: "warning" },
  4: { label: "超时", tag: "primary" },
};

/** LLM 调用状态展示 */
export const CALL_STATUS_META: Record<
  number,
  { label: string; tag: "success" | "danger" | "warning" }
> = {
  1: { label: "成功", tag: "success" },
  2: { label: "失败", tag: "danger" },
  3: { label: "超时", tag: "warning" },
};

/** 物理调用尝试状态：逐 Key/逐路由，1 成功 / 2 失败 / 3 跳过(熔断) */
export const ATTEMPT_STATUS_META: Record<
  number,
  { label: string; tag: "success" | "danger" | "warning" }
> = {
  1: { label: "成功", tag: "success" },
  2: { label: "失败", tag: "danger" },
  3: { label: "跳过", tag: "warning" },
};

/** 过程链类型展示（conversation 为主对话不标） */
export const TRACE_TYPE_META: Record<
  string,
  { label: string; tag: "primary" | "success" | "warning" | "info" }
> = {
  summary: { label: "摘要压缩", tag: "warning" },
  memory_extraction: { label: "记忆提取", tag: "success" },
  suggestion: { label: "建议推荐", tag: "info" },
  step_summary: { label: "步骤摘要", tag: "primary" },
};

/** 非 conversation 的过程链返回标签元数据，主对话返回 undefined 不渲染 */
export function traceTypeMeta(traceType?: string) {
  if (!traceType || traceType === "conversation") return undefined;
  return TRACE_TYPE_META[traceType] ?? { label: traceType, tag: "info" as const };
}

/** 上下文构成项类型中文标签 */
export const CONTEXT_ITEM_META: Record<
  string,
  { label: string; color: string }
> = {
  system: { label: "系统提示", color: "#409eff" },
  summary: { label: "历史摘要", color: "#67c23a" },
  history: { label: "会话历史", color: "#e6a23c" },
  memory: { label: "长期记忆", color: "#9254de" },
  retrieval: { label: "检索内容", color: "#f56c6c" },
  tools: { label: "工具清单", color: "#909399" },
};

/** 消息角色展示（标签文本 + 颜色类型） */
export const MESSAGE_ROLE_META: Record<
  string,
  { label: string; tag: "primary" | "success" | "warning" | "info" }
> = {
  system: { label: "系统", tag: "info" },
  user: { label: "用户", tag: "primary" },
  assistant: { label: "助手", tag: "success" },
  tool: { label: "工具", tag: "warning" },
};

export function fmtDuration(ms?: number) {
  if (ms == null) return "-";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function fmtTokens(tokens?: number) {
  if (tokens == null) return "-";
  return tokens.toLocaleString();
}
