// AI 对话无状态组件层：本地 VM（View Model）类型定义。
// 零外部依赖：不 import dehaze-sdk-js / pinia / vue，仅描述组件渲染所需的本地数据形态。
// wire → VM 的映射统一由 adapters/fromSdk.ts 负责，组件只消费本文件的类型、仅经 props/emit 通信。
// 约定：可选字段一律 `field?: T`（后端 null 字段不下发），仅语义确需空值时用 null（如 endAt）。

/** 数据范围：用户端会话 / 管理端会话审计 */
export type ChatScopeVM = "self" | "admin";

/** 消息状态（对应 wire status 数值：1-流式中，2-已完成，3-失败，4-已取消） */
export type ChatMessageStatusVM =
  "streaming" | "completed" | "failed" | "canceled";

/** 单段思考内容（后端按 content_block.start/stop 分段推送） */
export interface ChatThinkingSegmentVM {
  text: string;
  /** 是否已收到 content_block.stop */
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

/** 推理步骤（对应 wire ThoughtEvent / AiMessageThought，字段全 camelCase） */
export interface ChatThoughtStepVM {
  position: number;
  thought?: string;
  tool?: string;
  toolInput?: unknown;
  observation?: string;
  /** 步骤状态：1-成功，2-失败，3-跳过 */
  status: number;
  /** 失败原因（status=2 时填充） */
  error?: string;
  /** 本步骤耗时（毫秒） */
  latencyMs?: number;
}

/** 工具调用（name 可选，args 为已解析的入参或原始字符串） */
export interface ChatToolCallVM {
  name?: string;
  arguments?: unknown;
}

/** 中间产物 */
export interface ChatArtifactVM {
  id: number;
  type: string;
  summary?: unknown;
  /** 是否失效（关联文件被删除时标记） */
  invalid: boolean;
}

/** 长期记忆 */
export interface ChatMemoryVM {
  id: number;
  memoryType: string;
  content: string;
  source: string;
}

/** 消息反馈：rating 1-点赞，-1-点踩 */
export interface ChatFeedbackVM {
  rating: 1 | -1;
  tags?: string[];
  comment?: string;
}

/** 子智能体粒度用量（message.end usage.subAgents；仅在存在子智能体调用时下发） */
export interface ChatSubAgentUsageVM {
  /** 子智能体编码 */
  agentCode: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  /** 积分消耗 */
  credits: number;
}

/** Token / 积分用量 */
export interface ChatUsageVM {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  credits: number;
  /** 子智能体粒度用量（后端不下发时不落键，消费端据此不渲染空壳） */
  subAgents?: ChatSubAgentUsageVM[];
}

/** 中断推荐主算法（confirm） */
export interface ChatInterruptRecommendationVM {
  algorithmId: number;
  algorithmName: string;
  reason: string;
}

/** 中断备选算法（confirm） */
export interface ChatInterruptAlternativeVM {
  algorithmId: number;
  algorithmName: string;
  matchScore: number;
  reason: string;
}

/** 计划状态（plan 事件 status / 任务 status 归一） */
export type ChatPlanStatusVM = "pending" | "running" | "completed" | "failed";

/**
 * 计划任务项（plan 事件与 plan_approve 中断同形状，wire 与 VM 同名）。
 * 字段均可缺省：后端规划任务为渐进填充（首帧可能只有描述），组件按缺省容错展示。
 */
export interface ChatPlanTaskVM {
  id?: string;
  description?: string;
  dependsOn?: string[];
  status?: ChatPlanStatusVM;
  /** 任务范式（如 react / plan_execute） */
  paradigm?: string;
}

/** 计划修订记录（wire revisions 项；后端每次下发全量，revisionNo 恒有） */
export interface ChatPlanRevisionVM {
  revisionNo: number;
  reason: string;
}

/** 计划（Plan-and-Execute 展示态）：计划面板 / 中断卡共用 */
export interface ChatPlanVM {
  messageId?: number;
  /** 计划阶段（后端 phase 原样，如 planning / executing） */
  phase?: string;
  tasks: ChatPlanTaskVM[];
  status: ChatPlanStatusVM;
  revisions: ChatPlanRevisionVM[];
  /** 是否等待用户批准（plan_approve 中断激活时为 true） */
  awaitingApproval: boolean;
}

/** 新增任务载荷（plan_approve resume；对应 wire plan_edit.add 单对象） */
export interface ChatPlanAddTaskVM {
  description: string;
  dependsOn: string[];
}

/** 计划干预载荷（plan_approve resume；camelCase，由页面层映射为 wire 的 plan_edit） */
export interface ChatPlanEditVM {
  /** 需移除的任务 id */
  remove?: string[];
  /** 期望的任务 id 顺序 */
  reorder?: string[];
  /** 新增任务：后端 wire plan_edit.add 仅接受单对象，故一次只允许新增一条 */
  add?: ChatPlanAddTaskVM;
}

/** 中断确认子类型（confirm 中断细分；驱动 InterruptCard 路由渲染） */
export type ChatConfirmKindVM =
  "algorithm_recommend" | "tool_permission" | "dangerous_op" | "write_conflict";

/**
 * 推理中断展示字段（wire 与 VM 同为 camelCase，键名无需翻译）。
 * 适配层仅做集合筛选（推荐主结果/备选）与 confirmKind 的 write_conflict 派生。
 */
export interface ChatInterruptVM {
  type: "confirm" | "quota" | "async_wait" | "plan_approve";
  /** confirm：细分确认类型（算法推荐 / 工具授权 / 危险操作 / 写入冲突） */
  confirmKind?: ChatConfirmKindVM;
  /** confirm(write_conflict)：写冲突动作标识 */
  action?: "write_conflict";
  /** confirm：算法推荐主结果 */
  recommendation?: ChatInterruptRecommendationVM;
  /** confirm：算法推荐备选 */
  alternatives?: ChatInterruptAlternativeVM[];
  /** quota：升级引导文案 */
  upgradeTip?: string;
  /** quota：今日已用 / 日限额 */
  usedDaily?: number;
  dailyLimit?: number;
  /** quota：本月已用 / 月限额 */
  usedMonthly?: number;
  monthlyLimit?: number;
  /** async_wait：预计耗时 */
  estDuration?: string;
  /** async_wait：图片数量 */
  imageCount?: number;
  /** plan_approve：待确认计划任务 */
  plan?: ChatPlanTaskVM[];
  /** 其他中断：原因 / 说明 */
  reason?: string;
  detail?: string;
}

/**
 * 恢复中断载荷（wire MessageResumeForm 与 VM 同形，直接透传）。
 * planEdit 对应 wire plan_edit，两者均为 camelCase 单对象 add。
 */
export interface ChatResumeFormVM {
  /** confirm 中断：true 接受推荐，false 拒绝 */
  confirm?: boolean;
  /** 确认参数（如 algorithmId 表示选择了备选算法） */
  params?: Record<string, unknown>;
  /** plan_approve：计划干预 */
  planEdit?: ChatPlanEditVM;
}

/** 用户消息 */
export interface ChatUserMessageVM {
  role: "user";
  id: number;
  content: string;
  /** 是否被编辑过（重发后展示"已编辑"标识） */
  edited?: boolean;
  /** 编辑前原文（悬停查看） */
  originalContent?: string;
}

/** 助手消息：聚合思考过程 / 推理链 / 工具调用 / 产物 / 记忆 / 反馈 / 推荐问题 */
export interface ChatAssistantMessageVM {
  role: "assistant";
  id: number;
  status: ChatMessageStatusVM;
  text: string;
  error?: string;
  /** 思考过程（流式态优先，历史回退合成）；无思考为 null */
  thinking: ChatThinkingVM | null;
  /** 全部推理步骤（纯思考 + 工具步骤）；推理链展示时用 filterToolSteps 过滤 */
  steps: ChatThoughtStepVM[];
  toolCalls: ChatToolCallVM[];
  artifacts: ChatArtifactVM[];
  memories: ChatMemoryVM[];
  feedback: ChatFeedbackVM | null;
  usage?: ChatUsageVM;
  /** 推荐追问（仅最后一条 assistant 消息有意义） */
  suggestions: string[];
  /** 本次回复注入引用的长期记忆 ID 清单（wire usedMemoryIds 透传，供过程透明上下文构成派生） */
  usedMemoryIds?: number[];

  // ===== 以下为宿主绑定层注入的派生视图（非 wire 映射；无数据时不注入）=====
  /** 计划（Plan-and-Execute，来自 store.planByMessage） */
  plan?: ChatPlanVM;
  /** 分支视图（同一分叉点兄弟消息前端派生） */
  branch?: ChatBranchVM;
  /** 过程透明视图（步骤时间线 + 摘要 + 上下文构成，用户端"查看过程"） */
  process?: ChatProcessVM;
}

/**
 * 工具 / 内联步骤消息。
 * `role` 承载非 user/assistant 的内联步骤消息（wire role=system 归一），渲染为工具调用卡。
 */
export interface ChatToolMessageVM {
  role: "tool";
  id: number;
  /** 工具名（来源为首个工具调用的 name；无则为空） */
  name?: string;
  content: string;
  toolCalls: ChatToolCallVM[];
  /** 执行状态（wire status 归一；缺省不展示徽标） */
  status?: ChatMessageStatusVM;
}

/** 判别联合：按 role 分 user / assistant / tool 三支 */
export type ChatMessageVM =
  ChatUserMessageVM | ChatAssistantMessageVM | ChatToolMessageVM;

// ==================== 用户端过程透明（可观测性） ====================

/** 上下文构成标签（上下文组装占比可视化） */
export interface ChatContextChipVM {
  /** 构成类别：系统提示 / 历史 / 记忆 / 检索 / 工具清单 */
  kind: "system" | "history" | "memory" | "retrieval" | "tools";
  label: string;
  /** 占比（0-100） */
  ratio?: number;
  /** 来源详情（人类可读，不透出 raw 报文） */
  detail?: string;
}

/** 过程步骤（"查看过程"时间线单步；label/detail 均为人类可读，不透出 raw 报文） */
export interface ChatTraceStepVM {
  id: string;
  kind:
    "input" | "context" | "llm_call" | "tool_exec" | "system_event" | "billing";
  label: string;
  status: "ok" | "failed" | "running";
  error?: string;
  latencyMs?: number;
  detail?: string;
}

/** 过程摘要：步数 / 耗时 / 消耗，异常时显著标注失败环节 */
export interface ChatTraceSummaryVM {
  stepCount: number;
  durationMs?: number;
  credits?: number;
  failedCount?: number;
}

/**
 * 助手消息分支视图（宿主绑定层前端派生）。
 * 同一分叉点（parentMessageId 相同）的助手消息互为兄弟；`current` 从 1 计。
 */
export interface ChatBranchVM {
  current: number;
  total: number;
}

/** 用户端"查看过程"视图：步骤时间线 + 摘要 + 上下文构成（由助手消息真实字段派生，不透出 raw 报文） */
export interface ChatProcessVM {
  steps: ChatTraceStepVM[];
  summary: ChatTraceSummaryVM;
  chips: ChatContextChipVM[];
}
