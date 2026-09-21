// wire → VM 适配层：本目录唯一允许 import dehaze-sdk-js 的文件。
// 将 SSE 事件 / 接口返回的 wire 数据结构映射为本地 VM（见 ../types.ts）。
import type {
  AiMessageThought,
  AiMessageVO,
  ArtifactVO,
  FeedbackVO,
  InterruptData,
  InterruptEvent,
  MemoryVO,
  Plan,
  PlanPayload,
  PlanRevision,
  PlanTask,
  ThoughtEvent,
  TokenUsage,
} from "dehaze-sdk-js";
import type {
  ChatArtifactVM,
  ChatAssistantMessageVM,
  ChatConfirmKindVM,
  ChatFeedbackVM,
  ChatInterruptVM,
  ChatMemoryVM,
  ChatMessageStatusVM,
  ChatMessageVM,
  ChatPlanStatusVM,
  ChatPlanTaskVM,
  ChatPlanVM,
  ChatSubAgentUsageVM,
  ChatThinkingVM,
  ChatThoughtStepVM,
  ChatToolCallVM,
  ChatUsageVM,
} from "../types";

/** wire 子智能体粒度用量（message.end usage.subAgents 元素） */
type WireSubAgentUsage = NonNullable<TokenUsage["subAgents"]>[number];
import { buildThinkingFromThoughts } from "../vm/thinking";

/**
 * 流式思考态的结构化入参（对应 chat store 的 ThinkingState）。
 * 以结构化形状接收，避免适配层依赖 pinia store 的具体类型。
 */
export interface ThinkingStateLike {
  segments: { text: string; closed: boolean }[];
  startAt: number;
  endAt: number | null;
}

/** 推理步骤输入：流式 ThoughtEvent 与落库 AiMessageThought 的公共字段子集 */
type ThoughtInput = ThoughtEvent | AiMessageThought;

/** toMessageVM 的结构化入参（禁止长位置参数列表） */
export interface ToMessageVMOptions {
  message: AiMessageVO;
  /** 流式思考态（存在即优先，否则由 thoughts 合成） */
  thinking?: ThinkingStateLike | null;
  /** 推理步骤（流式 thought 事件或消息历史 thoughts），用于 steps 与思考回退合成 */
  thoughts?: readonly ThoughtInput[];
  /** 工具调用（流式草稿优先；缺省回退 message.toolCalls） */
  toolCalls?: unknown;
  /** 消息关联产物 */
  artifacts?: readonly ArtifactVO[];
  /** 回复引用的长期记忆 */
  memories?: readonly MemoryVO[];
  /** 消息反馈 */
  feedback?: FeedbackVO | null;
  /** 推荐追问 */
  suggestions?: readonly string[];
  /** 子智能体粒度用量（message.end usage.subAgents；缺省不落键，消费端不渲染空壳） */
  subAgents?: readonly WireSubAgentUsage[];
}

/** wire status（1/2/3/4）→ VM 状态；缺失/未知回退 completed（可渲染的终态） */
function toMessageStatusVM(status: number | undefined): ChatMessageStatusVM {
  switch (status) {
    case 1:
      return "streaming";
    case 2:
      return "completed";
    case 3:
      return "failed";
    case 4:
      return "canceled";
    default:
      return "completed";
  }
}

/** 工具调用列表归一：非数组按空列表处理，逐项容错取 name / arguments */
function normalizeToolCalls(raw: unknown): ChatToolCallVM[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    if (!item || typeof item !== "object") return {};
    const record = item as { name?: unknown; arguments?: unknown };
    return {
      ...(typeof record.name === "string" ? { name: record.name } : {}),
      ...("arguments" in record ? { arguments: record.arguments } : {}),
    };
  });
}

/** 推理步骤 → VM（字段缺失容错，仅透传存在的可选字段） */
function toThoughtStepVM(step: ThoughtInput): ChatThoughtStepVM {
  return {
    position: step.position,
    status: step.status,
    ...(step.thought !== undefined ? { thought: step.thought } : {}),
    ...(step.tool !== undefined ? { tool: step.tool } : {}),
    ...(step.toolInput !== undefined ? { toolInput: step.toolInput } : {}),
    ...(step.observation !== undefined
      ? { observation: step.observation }
      : {}),
    ...(step.error !== undefined ? { error: step.error } : {}),
    ...(step.latencyMs !== undefined ? { latencyMs: step.latencyMs } : {}),
  };
}

/**
 * 思考态映射：流式态优先（含累积计时的多段），无流式态时由推理步骤合成历史态。
 * 结构性入参 state/thoughts 任缺一即可，均无内容返回 null。
 */
export function toThinkingVM(input: {
  state?: ThinkingStateLike | null;
  thoughts?: readonly ThoughtInput[];
}): ChatThinkingVM | null {
  const state = input.state;
  if (state && state.segments.length > 0) {
    return {
      segments: state.segments.map((segment) => ({
        text: segment.text,
        closed: segment.closed,
      })),
      startAt: state.startAt,
      endAt: state.endAt,
      streaming: state.segments.some((segment) => !segment.closed),
    };
  }
  return buildThinkingFromThoughts((input.thoughts ?? []).map(toThoughtStepVM));
}

/** 子智能体粒度用量映射（wire 与 VM 同形，逐字段透传） */
function toSubAgentUsageVM(agent: WireSubAgentUsage): ChatSubAgentUsageVM {
  return {
    agentCode: agent.agentCode,
    inputTokens: agent.inputTokens,
    outputTokens: agent.outputTokens,
    cachedInputTokens: agent.cachedInputTokens,
    credits: agent.credits,
  };
}

/** 用量映射：全字段缺失返回 null；存在任一字段则其余缺省补 0；subAgents 仅非空时落键 */
export function toUsageVM(
  usage:
    | {
        inputTokens?: number;
        outputTokens?: number;
        cachedInputTokens?: number;
        credits?: number;
        subAgents?: readonly WireSubAgentUsage[] | null;
      }
    | null
    | undefined
): ChatUsageVM | null {
  if (!usage) return null;
  const { inputTokens, outputTokens, cachedInputTokens, credits, subAgents } =
    usage;
  if (
    inputTokens == null &&
    outputTokens == null &&
    cachedInputTokens == null &&
    credits == null
  ) {
    return null;
  }
  return {
    inputTokens: inputTokens ?? 0,
    outputTokens: outputTokens ?? 0,
    cachedInputTokens: cachedInputTokens ?? 0,
    credits: credits ?? 0,
    ...(subAgents && subAgents.length > 0
      ? { subAgents: subAgents.map(toSubAgentUsageVM) }
      : {}),
  };
}

/** 产物映射：wire isInvalid（0/1 数值）归一为 invalid 布尔 */
export function toArtifactVM(artifact: ArtifactVO): ChatArtifactVM {
  return {
    id: artifact.id,
    type: artifact.type,
    invalid: artifact.isInvalid === 1,
    ...(artifact.summary !== undefined ? { summary: artifact.summary } : {}),
  };
}

/** 记忆映射 */
export function toMemoryVM(memory: MemoryVO): ChatMemoryVM {
  return {
    id: memory.id,
    memoryType: memory.memoryType,
    content: memory.content,
    source: memory.source,
  };
}

/** 反馈映射：无反馈返回 null；rating 语义（1 点赞 / -1 点踩）原样保留 */
export function toFeedbackVM(
  feedback: FeedbackVO | null | undefined
): ChatFeedbackVM | null {
  if (!feedback) return null;
  return {
    rating: feedback.rating,
    ...(feedback.tags !== undefined ? { tags: feedback.tags } : {}),
    ...(feedback.comment !== undefined ? { comment: feedback.comment } : {}),
  };
}

/**
 * 读取 confirm 中断子类型（非 confirm / 未知 / 缺失均返回 undefined，不抛错）。
 * 展示映射不得因未知子类型崩溃；未知的拦截在 resume 路由处（store resolveConfirmKind）显式报错。
 * write_conflict 由 confirmKind=dangerous_op + action=write_conflict 派生。
 */
export function readConfirmKind(
  data: InterruptData | undefined
): ChatConfirmKindVM | undefined {
  const raw = data?.confirmKind;
  if (raw === "algorithm_recommend" || raw === "tool_permission") return raw;
  if (raw === "dangerous_op") {
    return data?.action === "write_conflict"
      ? "write_conflict"
      : "dangerous_op";
  }
  return undefined;
}

/** 中断映射：wire 与 VM 同为 camelCase，此处只做集合筛选（recommendation/alternatives）与缺省不落键 */
export function toInterruptVM(interrupt: InterruptEvent): ChatInterruptVM {
  // wire 可能缺省 data（后端仅下发 type），按空对象容错
  const data: InterruptData = interrupt.data ?? {};
  const planTasks = data.plan?.tasks;
  const confirmKind = readConfirmKind(interrupt.data);
  return {
    type: interrupt.type,
    ...(confirmKind !== undefined ? { confirmKind } : {}),
    ...(data.action === "write_conflict" ? { action: "write_conflict" } : {}),
    ...(data.recommendation
      ? {
          recommendation: {
            algorithmId: data.recommendation.algorithmId,
            algorithmName: data.recommendation.algorithmName,
            reason: data.recommendation.reason,
          },
        }
      : {}),
    ...(data.alternatives && data.alternatives.length > 0
      ? {
          alternatives: data.alternatives.map((alt) => ({
            algorithmId: alt.algorithmId,
            algorithmName: alt.algorithmName,
            matchScore: alt.matchScore,
            reason: alt.reason,
          })),
        }
      : {}),
    ...(data.upgradeTip !== undefined ? { upgradeTip: data.upgradeTip } : {}),
    ...(data.usedDaily !== undefined ? { usedDaily: data.usedDaily } : {}),
    ...(data.dailyLimit !== undefined ? { dailyLimit: data.dailyLimit } : {}),
    ...(data.usedMonthly !== undefined
      ? { usedMonthly: data.usedMonthly }
      : {}),
    ...(data.monthlyLimit !== undefined
      ? { monthlyLimit: data.monthlyLimit }
      : {}),
    ...(data.estDuration !== undefined
      ? { estDuration: data.estDuration }
      : {}),
    ...(data.imageCount !== undefined ? { imageCount: data.imageCount } : {}),
    ...(planTasks && planTasks.length > 0
      ? { plan: planTasks.map(toPlanTaskVM) }
      : {}),
    ...(data.reason !== undefined ? { reason: data.reason } : {}),
    ...(data.detail !== undefined ? { detail: data.detail } : {}),
  };
}

/**
 * 计划状态归一：后端 plan.status ∈ {pending, executing, revised, done}、
 * task.status ∈ {pending, done, failed}；未知值归"待执行"（展示映射不得抛错）。
 */
function toPlanStatusVM(raw: string): ChatPlanStatusVM {
  switch (raw) {
    // 重规划（revised）后计划仍在执行，展示上与 executing 同档
    case "executing":
    case "revised":
      return "running";
    case "done":
      return "completed";
    case "failed":
      return "failed";
    default:
      return "pending";
  }
}

/** 计划任务项归一：wire 已为 camelCase，仅保留 VM 展示字段（缺省不落键） */
function toPlanTaskVM(task: PlanTask): ChatPlanTaskVM {
  return {
    ...(task.id !== undefined ? { id: task.id } : {}),
    ...(task.description !== undefined
      ? { description: task.description }
      : {}),
    ...(task.dependsOn !== undefined ? { dependsOn: task.dependsOn } : {}),
    ...(task.status !== undefined
      ? { status: toPlanStatusVM(task.status) }
      : {}),
    ...(task.paradigm !== undefined ? { paradigm: task.paradigm } : {}),
  };
}

/** 计划归约选项（messageId/awaitingApproval 由调用方给定） */
export interface PlanVMOptions {
  messageId?: number;
  previous?: ChatPlanVM;
  awaitingApproval?: boolean;
}

/**
 * 计划 → ChatPlanVM 公共归约（plan 事件与 plan_approve 中断同形状）。传入 `previous` 时按前次计划累积——
 * 修订记录按 revisionNo 去重、升序（后端每次下发全量 revisions 且可能乱序），状态/阶段取最新事件，
 * 从而"生成/更新/重规划"演进可见、修订说明不丢失不重复。
 */
function buildPlanVM(
  tasks: ChatPlanTaskVM[],
  meta: { status?: string; revisions: PlanRevision[]; phase?: string },
  options: PlanVMOptions
): ChatPlanVM {
  const { previous } = options;
  const revisions = [...(previous?.revisions ?? [])];
  const seen = new Set(revisions.map((item) => item.revisionNo));
  for (const revision of meta.revisions) {
    if (seen.has(revision.revisionNo)) continue;
    seen.add(revision.revisionNo);
    revisions.push({
      revisionNo: revision.revisionNo,
      reason: revision.reason,
    });
  }
  revisions.sort((a, b) => a.revisionNo - b.revisionNo);

  const messageId = options.messageId ?? previous?.messageId;
  const status =
    meta.status !== undefined
      ? toPlanStatusVM(meta.status)
      : (previous?.status ?? "pending");
  const phase = meta.phase ?? previous?.phase;
  return {
    ...(messageId !== undefined ? { messageId } : {}),
    ...(phase !== undefined ? { phase } : {}),
    tasks,
    status,
    revisions,
    awaitingApproval:
      options.awaitingApproval ?? previous?.awaitingApproval ?? false,
  };
}

/** plan 事件 → ChatPlanVM（事件侧比中断侧多 phase） */
export function toPlanVM(plan: Plan, options: PlanVMOptions = {}): ChatPlanVM {
  return buildPlanVM(
    plan.tasks.map(toPlanTaskVM),
    { status: plan.status, revisions: plan.revisions, phase: plan.phase },
    options
  );
}

/** plan_approve 中断携带的计划 → ChatPlanVM（与事件侧同形，无 phase、无待批准之外的差异） */
export function toPlanVMFromInterrupt(
  plan: PlanPayload,
  options: PlanVMOptions = {}
): ChatPlanVM {
  return buildPlanVM(
    plan.tasks.map(toPlanTaskVM),
    { status: plan.status, revisions: plan.revisions },
    options
  );
}

/** 助手消息映射 */
function toAssistantMessageVM(
  options: ToMessageVMOptions
): ChatAssistantMessageVM {
  const { message, thoughts } = options;
  const usage = toUsageVM({
    inputTokens: message.inputTokens,
    outputTokens: message.outputTokens,
    cachedInputTokens: message.cachedInputTokens,
    credits: message.credits,
    subAgents: options.subAgents,
  });
  return {
    role: "assistant",
    id: message.id,
    status: toMessageStatusVM(message.status),
    text: message.content ?? "",
    thinking: toThinkingVM({ state: options.thinking, thoughts }),
    steps: (thoughts ?? []).map(toThoughtStepVM),
    toolCalls: normalizeToolCalls(options.toolCalls ?? message.toolCalls),
    artifacts: (options.artifacts ?? []).map(toArtifactVM),
    memories: (options.memories ?? []).map(toMemoryVM),
    feedback: toFeedbackVM(options.feedback),
    suggestions: [...(options.suggestions ?? [])],
    ...(message.error !== undefined ? { error: message.error } : {}),
    ...(usage !== null ? { usage } : {}),
  };
}

/**
 * 消息映射：按 wire role 分流（user / assistant / 其余归 tool）。
 * wire role=system 的第三类消息统一映射为 tool 分支渲染。
 */
export function toMessageVM(options: ToMessageVMOptions): ChatMessageVM {
  const { message } = options;
  if (message.role === "assistant") return toAssistantMessageVM(options);
  if (message.role === "user") {
    return {
      role: "user",
      id: message.id,
      content: message.content ?? "",
      ...(message.edited === 1 ? { edited: true } : {}),
      ...(message.originalContent != null
        ? { originalContent: message.originalContent }
        : {}),
    };
  }
  const toolCalls = normalizeToolCalls(options.toolCalls ?? message.toolCalls);
  return {
    role: "tool",
    id: message.id,
    content: message.content ?? "",
    toolCalls,
    status: toMessageStatusVM(message.status),
    ...(toolCalls[0]?.name !== undefined ? { name: toolCalls[0].name } : {}),
  };
}
