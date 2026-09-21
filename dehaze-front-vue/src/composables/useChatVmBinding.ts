// AI 对话宿主绑定层：把 store 的 wire 数据经 ai-chat 适配层（fromSdk）映射为无状态组件消费的 VM，
// 并把无状态组件的 emit 回接 store 方法。
// 用户端对话页（scope=self）与管理端会话审计抽屉（scope=admin）共用同一映射，避免两页各自重复实现；
// 组件层保持零 store / 零 SDK / 零 router / 零 ElMessage 的纯 props+emit 形态，耦合全部收敛在此。
import {
  AiConversationAPI,
  type AiMessageVO,
  type ArtifactVO,
} from "dehaze-sdk-js";
import { computed, ref, toValue, watch, type MaybeRefOrGetter } from "vue";
import {
  toInterruptVM,
  toMessageVM,
} from "@/components/ai-chat/adapters/fromSdk";
import type {
  ChatAssistantMessageVM,
  ChatBranchVM,
  ChatContextChipVM,
  ChatFeedbackVM,
  ChatInterruptVM,
  ChatMessageVM,
  ChatProcessVM,
  ChatResumeFormVM,
  ChatScopeVM,
  ChatTraceStepVM,
  ChatTraceSummaryVM,
} from "@/components/ai-chat/types";
import { useChatStore } from "@/store/modules/chat";

/** 助手消息分支视图：每个分叉点的选中态（forkKey → 选中的助手消息 id） */
type BranchSelection = Record<string, number>;

/** 同一分叉点键：parentMessageId 相同即同叉（无父归入 root） */
function forkKeyOf(message: AiMessageVO): string {
  return `b:${message.parentMessageId ?? "root"}`;
}

/**
 * 助手消息分支视图（渲染用，按 parentMessageId 分组）：后端消息列表返回会话全部消息
 * （含各分支，不作分支链过滤），故渲染层仍需按分叉点（parentMessageId 相同的助手消息
 * 互为兄弟）分组以决定展示哪一条；默认展示列表序最后一条（最新）。
 * 分支的真实切换由后端 getBranches/switchBranch 落库（见 onSwitchBranch），成功后回写本地选中态。
 * 返回需隐藏的兄弟索引与逐个索引的分支视图（current 从 1 计）。
 */
function deriveBranches(
  raw: readonly AiMessageVO[],
  selection: BranchSelection
): { hidden: Set<number>; byIndex: Map<number, ChatBranchVM> } {
  const groups = new Map<string, number[]>();
  raw.forEach((message, index) => {
    if (message.role !== "assistant") return;
    const key = forkKeyOf(message);
    const list = groups.get(key);
    if (list) list.push(index);
    else groups.set(key, [index]);
  });
  const hidden = new Set<number>();
  const byIndex = new Map<number, ChatBranchVM>();
  for (const [key, indexes] of groups) {
    if (indexes.length < 2) continue;
    const chosenId = selection[key];
    const chosen =
      chosenId !== undefined && indexes.some((i) => raw[i].id === chosenId)
        ? indexes.find((i) => raw[i].id === chosenId)!
        : indexes[indexes.length - 1];
    indexes.forEach((index, order) => {
      byIndex.set(index, { current: order + 1, total: indexes.length });
      if (index !== chosen) hidden.add(index);
    });
  }
  return { hidden, byIndex };
}

// ===== 用户端过程透明派生（VM 层纯归约，输入均为助手消息真实字段，不透出 raw 报文）=====
// 待下沉适配层：当前无用户端过程透明 wire 数据源，故在绑定层由消息 VM 派生；
// 若后端提供过程透明 wire 数据（trace 步骤/上下文构成），wire→VM 映射应下沉 adapters/fromSdk.ts。

/** 推理步骤状态（wire 1成功/2失败/3跳过）→ 时间线状态（跳过归入正常，不虚构失败） */
function toTraceStatus(status: number): ChatTraceStepVM["status"] {
  return status === 2 ? "failed" : "ok";
}

/**
 * 由消息推理步骤派生步骤时间线：工具步骤归 tool_exec、纯思考步骤归 llm_call，
 * 计费节点由 usage 派生（六类 kind 中 input/context/system_event 用户端无数据源，不虚构）。
 */
function deriveTraceSteps(message: ChatAssistantMessageVM): ChatTraceStepVM[] {
  const steps: ChatTraceStepVM[] = message.steps
    .slice()
    .sort((a, b) => a.position - b.position)
    .map((step) => {
      const detail = step.observation ?? step.thought;
      return {
        id: `${message.id}:${step.position}`,
        kind: step.tool ? "tool_exec" : "llm_call",
        label: step.tool ?? "模型思考",
        status: toTraceStatus(step.status),
        ...(step.error !== undefined ? { error: step.error } : {}),
        ...(step.latencyMs !== undefined ? { latencyMs: step.latencyMs } : {}),
        ...(detail !== undefined ? { detail } : {}),
      } satisfies ChatTraceStepVM;
    });
  const usage = message.usage;
  if (
    usage &&
    (usage.credits > 0 || usage.inputTokens > 0 || usage.outputTokens > 0)
  ) {
    const cached =
      usage.cachedInputTokens > 0 ? `（缓存 ${usage.cachedInputTokens}）` : "";
    steps.push({
      id: `${message.id}:billing`,
      kind: "billing",
      label: "计费",
      status: "ok",
      detail: `输入 ${usage.inputTokens} / 输出 ${usage.outputTokens} tokens${cached} · 积分 ${usage.credits}`,
    });
  }
  return steps;
}

/** 过程摘要：步数 / 步骤耗时合计 / 积分消耗 / 失败步数（均由真实字段聚合，不虚构） */
function deriveTraceSummary(
  message: ChatAssistantMessageVM,
  steps: ChatTraceStepVM[]
): ChatTraceSummaryVM {
  const latencySum = steps.reduce((sum, s) => sum + (s.latencyMs ?? 0), 0);
  return {
    stepCount: steps.length,
    ...(latencySum > 0 ? { durationMs: latencySum } : {}),
    ...(message.usage ? { credits: message.usage.credits } : {}),
    failedCount: steps.filter((s) => s.status === "failed").length,
  };
}

/**
 * 上下文构成：仅用真实可得的 wire 字段（本轮注入引用的长期记忆 usedMemoryIds）。
 * 系统提示/历史/检索构成项用户端无数据源（属管理端 raw 口径），不虚构占位。
 */
function deriveContextChips(
  message: ChatAssistantMessageVM
): ChatContextChipVM[] {
  const ids = message.usedMemoryIds ?? [];
  if (ids.length === 0) return [];
  return [
    {
      kind: "memory",
      label: "记忆",
      detail: `本次回复注入引用的长期记忆共 ${ids.length} 条`,
    },
  ];
}

/** 由助手消息派生过程透明视图；无步骤且无上下文构成时不注入（不产生空壳） */
function deriveProcess(
  message: ChatAssistantMessageVM
): ChatProcessVM | undefined {
  const steps = deriveTraceSteps(message);
  const chips = deriveContextChips(message);
  if (steps.length === 0 && chips.length === 0) return undefined;
  return { steps, summary: deriveTraceSummary(message, steps), chips };
}

export interface ChatVmBindingOptions {
  /** 数据范围：用户端 self / 管理端审计 admin */
  scope: ChatScopeVM;
  /** wire 消息源（self：chatStore.messages；admin：adminAudit.detailMessages） */
  messages: MaybeRefOrGetter<AiMessageVO[]>;
  /** 是否还有更早历史（self 默认取 store，admin 由页面按分页口径给定） */
  hasMoreHistory?: MaybeRefOrGetter<boolean>;
  /** 更早历史加载中（self 默认取 store，admin 由页面给定） */
  loadingMore?: MaybeRefOrGetter<boolean>;
  /** 到达顶部加载更早历史（admin 由页面给定，如 loadMoreDetailMessages） */
  onLoadMore?: () => void;
  /** 链路下钻（管理端），self 无入口 */
  onTrace?: (message: ChatMessageVM) => void;
}

/**
 * 宿主绑定层：返回无状态组件所需的 VM props 与 emit 回接处理器。
 * 管理端（admin）显式使用 null/true 语义，不读取 store 的流式/滚动状态，杜绝 scope 语义泄漏。
 */
export function useChatVmBinding(options: ChatVmBindingOptions) {
  const chatStore = useChatStore();
  const selfScope = options.scope === "self";

  /** 消息产物：由组件 emit load-artifacts 驱动按需拉取后回填（组件层零 SDK） */
  const artifactsByMessage = ref<Record<number, ArtifactVO[]>>({});

  /** 助手消息分支选中态（forkKey → 选中助手消息 id）；仅用户端，切换不写 store */
  const activeBranchByFork = ref<BranchSelection>({});

  const messages = computed<ChatMessageVM[]>(() => {
    const raw = toValue(options.messages);
    const branches = selfScope
      ? deriveBranches(raw, activeBranchByFork.value)
      : null;
    const result: ChatMessageVM[] = [];
    raw.forEach((message, index) => {
      if (branches?.hidden.has(index)) return;
      const vm = toMessageVM({
        message,
        // 用户端：流式态/推理步骤/工具调用/记忆/反馈/推荐问题全部取自 store；
        // 管理端：仅消费 wire 自带推理步骤（只读审计，无本地流式态）
        ...(selfScope
          ? {
              thinking: chatStore.thinkingByMessage[message.id] ?? null,
              thoughts:
                chatStore.thoughtsByMessage[message.id] ??
                message.thoughts ??
                [],
              toolCalls:
                chatStore.toolCallsByMessage[message.id] ?? message.toolCalls,
              memories: chatStore.messageMemories[message.id] ?? [],
              feedback: chatStore.feedbackByMessage[message.id] ?? null,
              suggestions: chatStore.suggestions,
              subAgents: chatStore.subAgentsByMessage[message.id],
            }
          : { thoughts: message.thoughts ?? [] }),
        artifacts: artifactsByMessage.value[message.id] ?? [],
      });
      // 用户/工具消息的 wire 映射（edited/originalContent/status）已全部下沉适配层，此处直接透传
      if (vm.role !== "assistant") {
        result.push(vm);
        return;
      }
      // 助手消息：注入 wire 透传字段 + 宿主派生的计划/分支/过程透明视图（仅用户端）
      const assistant: ChatAssistantMessageVM = {
        ...vm,
        ...(message.usedMemoryIds?.length
          ? { usedMemoryIds: message.usedMemoryIds }
          : {}),
      };
      if (!selfScope) {
        result.push(assistant);
        return;
      }
      const plan = chatStore.planByMessage[message.id];
      const branch = branches?.byIndex.get(index);
      const process = deriveProcess(assistant);
      result.push({
        ...assistant,
        ...(plan ? { plan } : {}),
        ...(branch ? { branch } : {}),
        ...(process ? { process } : {}),
      });
    });
    return result;
  });

  const streamingMessageId = computed(() =>
    selfScope ? chatStore.streamingMessageId : null
  );
  const interruptedMessageId = computed(() =>
    selfScope ? chatStore.interruptedMessageId : null
  );
  const interrupts = computed<ChatInterruptVM[]>(() =>
    selfScope ? chatStore.interrupts.map(toInterruptVM) : []
  );
  const scrollFollowEnabled = computed(() =>
    selfScope ? chatStore.scrollFollowEnabled : true
  );
  const hasMoreHistory = computed(() =>
    options.hasMoreHistory !== undefined
      ? toValue(options.hasMoreHistory)
      : selfScope && chatStore.messagesHasMore
  );
  const loadingMore = computed(() =>
    options.loadingMore !== undefined
      ? toValue(options.loadingMore)
      : selfScope && chatStore.messagesLoadingMore
  );
  // 分支切换进行中：仅用户端有意义（管理端无切换入口，恒 false）
  const branchSwitching = computed(() =>
    selfScope ? chatStore.branchSwitching : false
  );

  // ===== 产物按需拉取（同一消息只拉一次；失败允许重试）=====
  // 用户端与管理端审计均需展示产物卡片（审计只读视角同样回放产物），故不按 scope 裁剪。
  const requestedArtifacts = new Set<number>();
  async function loadArtifacts(messageId: number) {
    if (messageId <= 0 || requestedArtifacts.has(messageId)) return;
    requestedArtifacts.add(messageId);
    try {
      const list = await AiConversationAPI.getMessageArtifacts(messageId);
      artifactsByMessage.value = {
        ...artifactsByMessage.value,
        [messageId]: list,
      };
    } catch {
      // 产物加载失败不阻塞消息展示
      requestedArtifacts.delete(messageId);
    }
  }

  // 用户端进入会话后为既有助手消息补拉反馈（等价旧组件"挂载即查询"，反馈渲染需落库态而非仅本次提交）
  if (selfScope) {
    const requestedFeedback = new Set<number>();
    watch(
      () => toValue(options.messages),
      (list) => {
        for (const message of list) {
          if (
            message.role === "assistant" &&
            message.id > 0 &&
            !requestedFeedback.has(message.id)
          ) {
            requestedFeedback.add(message.id);
            void chatStore.fetchFeedback(message.id);
          }
        }
      },
      { immediate: true }
    );
  }

  /** 按 VM id 反查 wire 消息（store 的引用/朗读等操作消费 wire 对象） */
  function wireOf(vm: ChatMessageVM) {
    return toValue(options.messages).find((item) => item.id === vm.id);
  }

  // ===== 组件 emit → store 回接 =====
  function onScrollFollowToggle(enabled: boolean) {
    if (selfScope) chatStore.scrollFollowEnabled = enabled;
  }

  function onReachTop() {
    if (selfScope) void chatStore.loadMoreMessages();
    else options.onLoadMore?.();
  }

  function onRegenerate(message: ChatMessageVM) {
    if (selfScope) chatStore.regenerate(message.id);
  }

  /** 失败态"重试"等价重新生成（同一消息分支） */
  function onRetry(message: ChatMessageVM) {
    if (selfScope) chatStore.regenerate(message.id);
  }

  function onQuote(message: ChatMessageVM) {
    const wire = wireOf(message);
    if (selfScope && wire) chatStore.quoteMessage(wire);
  }

  function onSpeak(message: ChatMessageVM) {
    const wire = wireOf(message);
    if (selfScope && wire) void chatStore.speakMessage(wire);
  }

  function onFeedback(
    message: ChatAssistantMessageVM,
    data: ChatFeedbackVM | null
  ) {
    if (selfScope) void chatStore.submitFeedback(message.id, data);
  }

  function onApplySuggestion(question: string) {
    if (selfScope) chatStore.applySuggestion(question);
  }

  function onResumeInterrupt(messageId: number, data: ChatResumeFormVM) {
    if (selfScope) chatStore.resumeInterrupt(messageId, data);
  }

  function onLoadArtifacts(messageId: number) {
    void loadArtifacts(messageId);
  }

  /**
   * 切换分支：以分叉点（父消息）为键向后端拉取真实兄弟分支（后端按时间倒序，反转为展示正序），
   * 按序号定位目标后调 switchBranch 落库（更新会话 currentBranchMessageId）并刷新消息列表；
   * 成功后记入本地选中态。失败给用户可见提示、保持原状态（不预置选中态）。管理端审计无此入口。
   */
  async function onSwitchBranch(message: ChatMessageVM, index: number) {
    if (!selfScope) return;
    const conversationId = chatStore.currentConversationId;
    const current = toValue(options.messages).find(
      (item) => item.id === message.id
    );
    const forkId = current?.parentMessageId;
    if (!conversationId || !current || !forkId) return;
    try {
      const siblings = await AiConversationAPI.getBranches(
        conversationId,
        forkId
      );
      const target = siblings.slice().reverse()[index - 1];
      if (!target) return;
      await chatStore.switchBranch(target.id);
      activeBranchByFork.value = {
        ...activeBranchByFork.value,
        [forkKeyOf(current)]: target.id,
      };
    } catch {
      ElMessage.error("切换分支失败，请稍后重试");
    }
  }

  function onTrace(message: ChatMessageVM) {
    options.onTrace?.(message);
  }

  return {
    messages,
    streamingMessageId,
    interruptedMessageId,
    interrupts,
    scrollFollowEnabled,
    hasMoreHistory,
    loadingMore,
    branchSwitching,
    loadArtifacts,
    onScrollFollowToggle,
    onReachTop,
    onSwitchBranch,
    onRegenerate,
    onRetry,
    onQuote,
    onSpeak,
    onFeedback,
    onApplySuggestion,
    onResumeInterrupt,
    onLoadArtifacts,
    onTrace,
  };
}
