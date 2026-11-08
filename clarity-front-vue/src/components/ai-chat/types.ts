// AI 对话无状态组件层的纯 UI 派生类型。
//
// 消息、块、计划、中断、用量、产物、反馈一律直接使用契约类型（import type 自 clarity-sdk-js），
// 本文件只保留"契约里没有、但渲染确实需要"的少量派生形状：思考分段与计时、工具步骤投影、
// 分支序号、组件动作与恢复表单。不做字段改名、不保留契约字段副本。
import type { FeedbackForm, MessageResume, ToolStatus } from "clarity-sdk-js";

/** 数据范围：用户端会话 / 管理端会话审计 */
export type ChatScopeVM = "self" | "admin";

/** 单段思考内容（契约 thinking 块按 seq 分段，块收尾即闭合） */
export interface ChatThinkingSegmentVM {
  text: string;
  /** 该思考块是否已收尾（契约 block.durationMs 非空） */
  closed: boolean;
}

/** 思考过程：多段思考 + 前端计时 */
export interface ChatThinkingVM {
  segments: ChatThinkingSegmentVM[];
  /** 首段开始时刻（前端时钟） */
  startAt: number;
  /** 末段结束时刻；null 表示仍在思考 */
  endAt: number | null;
  /** 是否仍在流式思考（存在未闭合段） */
  streaming: boolean;
}

/** 工具步骤（由契约 tool_use 块投影，status 直接用契约 ToolStatus） */
export interface ChatThoughtStepVM {
  position: number;
  tool?: string;
  toolInput?: unknown;
  observation?: string;
  status: ToolStatus;
  error?: string;
  latencyMs?: number;
}

/**
 * 助手消息分支视图（宿主绑定层前端派生）。
 * 同一分叉点（parentMessageId 相同）的助手消息互为兄弟；`current` 从 1 计。
 */
export interface ChatBranchVM {
  current: number;
  total: number;
}

/** 新增任务载荷（plan_approve resume；对应 wire plan_edit.add 单对象） */
export type ChatPlanAddTaskVM = {
  description: string;
  dependsOn: string[];
};

/** 计划干预载荷（plan_approve resume；camelCase，宿主原样透传给契约 MessageResume.planEdit） */
export type ChatPlanEditVM = {
  /** 需移除的任务 id */
  remove?: string[];
  /** 期望的任务 id 顺序 */
  reorder?: string[];
  /** 新增任务：后端 wire plan_edit.add 仅接受单对象，故一次只允许新增一条 */
  add?: ChatPlanAddTaskVM;
};

/**
 * 恢复中断表单（组件层构建、原样交给宿主回接 store）。
 * 结构与契约 MessageResume 兼容：planEdit 对应其 planEdit 字段。
 */
export interface ChatResumeFormVM {
  /** confirm 中断：true 接受推荐，false 拒绝 */
  confirm?: boolean;
  /** 确认参数（如 modelId 表示选择了备选模型） */
  params?: Record<string, unknown>;
  /** plan_approve：计划干预 */
  planEdit?: ChatPlanEditVM;
}

/**
 * 消息动作（单一 `action` 事件载荷，判别联合）。
 *
 * 组件层只上抛该联合，不再为每个动作定义独立 emit；copy/edit 由页面层就地处理
 * （剪贴板、编辑弹窗），其余动作由宿主绑定层回接 store（见 `dispatchMessageAction`）。
 */
export type ChatMessageActionVM =
  /** 复制消息文本（页面层处理） */
  | { type: "copy"; messageId: number }
  /** 编辑用户消息（页面层打开编辑框后调用 store.editMessage） */
  | { type: "edit"; messageId: number }
  /** 重新生成助手回复 */
  | { type: "regenerate"; messageId: number }
  /** 失败重试（等价重新生成） */
  | { type: "retry"; messageId: number }
  /** 引用消息后发送 */
  | { type: "quote"; messageId: number }
  /** 朗读助手回复 */
  | { type: "speak"; messageId: number }
  /** 删除消息 */
  | { type: "delete"; messageId: number }
  /** 提交/撤销反馈（data=null 表示撤销） */
  | { type: "feedback"; messageId: number; data: FeedbackForm | null }
  /** 恢复中断（载荷与契约 MessageResume 同形） */
  | { type: "resume-interrupt"; messageId: number; data: MessageResume }
  /** 采用推荐追问 */
  | { type: "apply-suggestion"; question: string };
