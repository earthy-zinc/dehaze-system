// 用户端会话视图：把 store 的契约 Message 原样交给无状态组件，并把组件事件回接 store。
//
// 后端的 self 消息列表已是当前激活链（分支选择在服务端），本层只做三件事：
//   1) 由消息自身的 siblingIndex/siblingCount 派生分支序号；
//   2) 进入会话后一次性批量拉取产物与反馈（流式新消息终态补单条产物）；
//   3) 把消息动作 / 滚动 / 分支切换转交 store。
// 不做分支选择、不过滤消息、不复制消息字段、不维护第二套流式或中断状态。
import { AiConversationAPI, type Artifact, type Message } from "clarity-sdk-js";
import { computed, ref, watch } from "vue";
import type { ChatBranchVM } from "@/components/ai-chat/types";
import { useChatStore } from "@/store/modules/chat";
import type { MessageAction } from "@/store/modules/chat/shared";

/** 会话产物批量拉取的分页大小（服务端上限 100） */
const ARTIFACTS_PAGE_SIZE = 100;

export function useChatSessionView() {
  const chatStore = useChatStore();

  // ===== 消息与分支 =====
  /** 激活链即后端返回的消息列表，前端不再做任何可见性过滤（不复制数组） */
  const messages = computed(() => chatStore.messageList);

  /** 分支序号对象按「消息 id + current/total」复用：流式帧重建 Record 时不换子组件 props 引用 */
  const branchCache = new Map<string, ChatBranchVM>();
  function branchVM(id: number, current: number, total: number): ChatBranchVM {
    const key = `${id}:${current}/${total}`;
    let vm = branchCache.get(key);
    if (!vm) {
      vm = { current, total };
      branchCache.set(key, vm);
    }
    return vm;
  }
  /** 分支序号只认消息自身字段：后端仅在 assistant 且存在分叉时填充 */
  const branches = computed<Record<number, ChatBranchVM>>(() => {
    const result: Record<number, ChatBranchVM> = {};
    for (const message of messages.value) {
      const current = message.siblingIndex;
      const total = message.siblingCount;
      if (message.role !== "assistant") continue;
      if (current == null || total == null || total <= 1) continue;
      result[message.id] = branchVM(message.id, current, total);
    }
    return result;
  });

  /** 会话切换竞态：请求发出后会话已变则丢弃回包 */
  function isStale(conversationId: number) {
    return chatStore.currentConversationId !== conversationId;
  }

  // ===== 产物：进入会话批量拉取，流式新消息终态补单条 =====
  const artifacts = ref<Record<number, Artifact[]>>({});
  /** 单条路径已请求过的消息 id（失败允许下次重试） */
  const requestedArtifacts = new Set<number>();
  /** 批量加载中的会话：期间挂载的消息先记账，批量落地后统一判断 */
  let artifactsBatchConversationId: number | null = null;
  /**
   * 批量结果覆盖范围（会话 + 发起批量时已存在的消息 id）。
   * 只按"是否在批量快照内"判断：批量在途期间才完成的轮次，其产物不在这次响应里，
   * 即使 id 不大也必须在批量落地后走单条补齐。
   */
  let artifactsCoverage: {
    conversationId: number;
    messageIds: Set<number>;
  } | null = null;
  /** 批量加载期间被跳过的消息（按会话分组）：批量落地后按需补齐 */
  const deferredArtifacts = new Map<number, Set<number>>();

  async function onLoadArtifacts(messageId: number) {
    if (messageId <= 0 || requestedArtifacts.has(messageId)) return;
    const conversationId = chatStore.currentConversationId;
    if (conversationId === null) return;
    if (artifactsBatchConversationId !== null) {
      const pending =
        deferredArtifacts.get(conversationId) ?? new Set<number>();
      pending.add(messageId);
      deferredArtifacts.set(conversationId, pending);
      return;
    }
    // 已被会话批量结果覆盖（批量发起时已加载的消息）
    if (
      artifactsCoverage?.conversationId === conversationId &&
      artifactsCoverage.messageIds.has(messageId)
    ) {
      return;
    }
    requestedArtifacts.add(messageId);
    try {
      const list = await AiConversationAPI.getMessageArtifacts(messageId);
      // 回包时会话已切走：丢弃，避免写进新会话的产物表
      if (chatStore.currentConversationId !== conversationId) return;
      artifacts.value = { ...artifacts.value, [messageId]: list };
    } catch {
      // 产物加载失败不阻塞消息展示
      requestedArtifacts.delete(messageId);
    }
  }

  /** 批量落地后补齐期间新出现、且不在批量覆盖范围内的消息 */
  function flushDeferredArtifacts(conversationId: number) {
    const ids = [...(deferredArtifacts.get(conversationId) ?? [])];
    deferredArtifacts.delete(conversationId);
    if (chatStore.currentConversationId !== conversationId) return;
    for (const id of ids) void onLoadArtifacts(id);
  }

  /**
   * 会话级批量产物：按页取全量（pageSize 上限 100，total 未取满继续翻页），
   * 按 artifact.messageId 归组。切换会话时整体替换；回包前校验会话未变。
   */
  async function loadConversationArtifacts(
    conversationId: number
  ): Promise<boolean> {
    artifactsBatchConversationId = conversationId;
    artifacts.value = {};
    // 覆盖范围取发起请求时的消息快照：批量响应之后才出现的轮次不在这次快照内，
    // 其产物必须留给单条路径补齐（不能按 id 大小推断"已被批量覆盖"）。
    const coveredMessageIds = new Set(
      chatStore.messageList.map((message) => message.id)
    );
    const collected: Artifact[] = [];
    try {
      let pageNum = 1;
      while (true) {
        const page = await AiConversationAPI.getArtifacts(conversationId, {
          pageNum,
          pageSize: ARTIFACTS_PAGE_SIZE,
        });
        if (isStale(conversationId)) return false;
        const list = page.list ?? [];
        collected.push(...list);
        if (list.length === 0 || collected.length >= page.total) break;
        pageNum += 1;
      }
    } catch {
      // 产物加载失败不阻塞消息展示（挂载消息退回单条路径按需补）
      return false;
    } finally {
      if (artifactsBatchConversationId === conversationId) {
        artifactsBatchConversationId = null;
      }
    }
    if (isStale(conversationId)) return false;
    const grouped: Record<number, Artifact[]> = {};
    for (const artifact of collected) {
      (grouped[artifact.messageId] ??= []).push(artifact);
    }
    artifacts.value = grouped;
    artifactsCoverage = { conversationId, messageIds: coveredMessageIds };
    return true;
  }

  // ===== 反馈：进入会话后一次批量回填（反馈是独立资源） =====
  async function loadConversationFeedback(
    conversationId: number
  ): Promise<boolean> {
    // 以发起请求时的本会话消息为范围：期间新出现的消息不在快照内，不能据空结果清掉其反馈
    const messageIds = chatStore.messageList.map((message) => message.id);
    try {
      const list =
        await AiConversationAPI.getConversationFeedback(conversationId);
      if (isStale(conversationId)) return false;
      chatStore.applyConversationFeedback(list ?? [], messageIds);
      return true;
    } catch {
      // 反馈查询失败按无反馈展示
      return false;
    }
  }

  /**
   * 进入会话（消息列表首次加载完成）后拉取本会话附加数据。
   * 两项都成功才记为已附加：任一失败（如首屏重试）会在下一次加载完成时重新拉取；
   * 会话切换后重置，重新进入同一会话也会刷新。
   */
  let attachedConversationId: number | null = null;
  let lastConversationId: number | null = null;
  watch(
    () => [chatStore.currentConversationId, chatStore.messagesLoading] as const,
    ([conversationId, loading]) => {
      if (conversationId !== lastConversationId) {
        lastConversationId = conversationId;
        attachedConversationId = null;
      }
      if (!conversationId || loading) return;
      if (attachedConversationId === conversationId) return;
      void (async () => {
        const results = await Promise.all([
          loadConversationArtifacts(conversationId),
          loadConversationFeedback(conversationId),
        ]);
        // 批量成功与失败都要回放：成功时未覆盖的消息走单条，失败时整批回退单条路径
        flushDeferredArtifacts(conversationId);
        if (
          results.every(Boolean) &&
          chatStore.currentConversationId === conversationId
        ) {
          attachedConversationId = conversationId;
        }
      })();
    },
    { immediate: true }
  );

  // ===== 流式 / 中断：激活链上的待处理中断 =====
  const pending = computed(() => {
    const list = messages.value;
    for (let index = list.length - 1; index >= 0; index--) {
      const message = list[index];
      if (
        message.pendingInterrupt &&
        message.pendingInterrupt.actionable !== false
      ) {
        return message;
      }
    }
    return null;
  });
  const isStreaming = computed(
    () => chatStore.streamStatus !== "idle" || pending.value !== null
  );
  const interruptedMessageId = computed<number | null>(() =>
    chatStore.streamStatus === "interrupted"
      ? chatStore.streamMessageId
      : (pending.value?.id ?? null)
  );

  // ===== 滚动与分页 =====
  const scrollFollowEnabled = computed(() => chatStore.scrollFollowEnabled);
  const hasMoreHistory = computed(() => chatStore.messagesHasMore);
  const loadingMore = computed(() => chatStore.messagesLoadingMore);
  const messagesError = computed(() => chatStore.messagesError);
  /**
   * 分支切换禁用态：切换进行中防重入；流式/中断期间禁止切换——
   * running 时切换会把正在生成的分支藏起来，中断（含刷新后只剩持久化 actionable
   * 中断、streamStatus 已 idle 的情况）时后端仍持有流锁，切换会被拒绝。
   */
  const branchDisabled = computed(
    () =>
      chatStore.branchSwitching ||
      chatStore.streamStatus !== "idle" ||
      pending.value !== null
  );

  function onScrollFollowToggle(enabled: boolean) {
    chatStore.scrollFollowEnabled = enabled;
  }

  function onReachTop() {
    void chatStore.loadMoreMessages();
  }

  /** 历史加载失败重试：错误已入 store 状态，失败不在此再提示 */
  function onRetryMessages() {
    void chatStore.retryMessages();
  }

  /**
   * 切换分支：兄弟分支可能不在激活链上，必须向服务端取分叉点的全部兄弟再按序号定位；
   * store 落库指针并重新拉取消息列表，切换后的展示以服务端返回的激活链为准。
   */
  async function onSwitchBranch(message: Message, index: number) {
    const conversationId = chatStore.currentConversationId;
    const forkId = message.parentMessageId;
    if (!conversationId || !forkId || branchDisabled.value) return;
    try {
      const siblings = await AiConversationAPI.getBranches(
        conversationId,
        forkId
      );
      // 后端按时间倒序返回，翻转为展示正序后按序号取目标
      const target = siblings.slice().reverse()[index - 1];
      if (!target) return;
      await chatStore.switchBranch(target.id);
    } catch {
      ElMessage.error("切换分支失败，请稍后重试");
    }
  }

  async function onMessageAction(action: MessageAction) {
    try {
      await chatStore.dispatchMessageAction(action);
    } catch {
      // 业务错误与网络错误已由请求层按后端 msg/网络原因提示，这里只负责吞掉 Promise
    }
  }

  return {
    messages,
    branches,
    artifacts,
    feedback: computed(() => chatStore.feedbackByMessage),
    isStreaming,
    interruptedMessageId,
    scrollFollowEnabled,
    hasMoreHistory,
    loadingMore,
    messagesError,
    branchDisabled,
    onScrollFollowToggle,
    onReachTop,
    onRetryMessages,
    onSwitchBranch,
    onMessageAction,
    onLoadArtifacts,
  };
}
