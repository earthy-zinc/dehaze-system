// 消息 slice：消息列表加载与消息级操作
import { AiConversationAPI, type AiMessageVO } from "dehaze-sdk-js";
import { ref } from "vue";
import { MESSAGES_PAGE_SIZE, nextLocalMessageId, type ChatCtx } from "./shared";

/** 在当前消息列表中按 id 查找消息（stream slice 归约时引用） */
export function findMessage(ctx: ChatCtx, messageId: number) {
  return ctx.messages.value.find((item) => item.id === messageId);
}

/** 追加本地 assistant 占位消息（负数 id，待 message.start 绑定服务端 id） */
export function appendAssistantPlaceholder(
  ctx: ChatCtx,
  conversationId: number,
  parentMessageId?: number
) {
  const placeholder: AiMessageVO = {
    id: nextLocalMessageId(),
    conversationId,
    role: "assistant",
    content: "",
    status: 1,
    parentMessageId,
    createTime: new Date().toISOString(),
  };
  ctx.messages.value.push(placeholder);
  return placeholder;
}

export function createMessagesSlice(ctx: ChatCtx, teardownStream: () => void) {
  const {
    messages,
    currentConversationId,
    scope,
    conversations,
    thoughtsByMessage,
  } = ctx;

  const messagesLoading = ref(false);
  const messagesLoadingMore = ref(false);
  const messagesHasMore = ref(false);
  /** 分支切换进行中：禁用切换器并防重复提交 */
  const branchSwitching = ref(false);

  /** 分页请求期间切换会话时丢弃过期结果（避免把 A 会话的历史拼到 B 会话） */
  function isStale(conversationId: number) {
    return currentConversationId.value !== conversationId;
  }

  /** 写入列表接口附带的历史思考链 */
  function collectThoughts(list: readonly AiMessageVO[]) {
    for (const m of list) {
      if (m.role === "assistant" && m.thoughts?.length) {
        thoughtsByMessage.value[m.id] = m.thoughts;
      }
    }
  }

  async function fetchMessages(conversationId: number) {
    if (ctx.session && ctx.session.conversationId !== conversationId) {
      teardownStream();
    }
    currentConversationId.value = conversationId;
    messagesLoading.value = true;
    try {
      // 首屏：缺省 before 取最新一页（游标分页按 id 倒序返回）
      const result = await AiConversationAPI.getMessages(conversationId, {
        limit: MESSAGES_PAGE_SIZE,
      });
      if (isStale(conversationId)) return;
      const descList = result.list ?? [];
      // 后端游标分页按 id 倒序返回，反转回时间正序展示（最早在上）
      const list = descList.slice().reverse();
      collectThoughts(descList);
      // hasMore 由响应直接给定（是否存在 id 小于本页最小 id 的消息）
      messagesHasMore.value = result.hasMore;
      // 流式进行中（同一会话）：保留当前流式消息对象，避免整体覆盖后
      // findMessage 找不到消息（content 不渲染、status 不清，界面永久"正在思考"）
      const live =
        ctx.session?.conversationId === conversationId
          ? ctx.session.message
          : null;
      if (live) {
        const idx = list.findIndex((m) => m.id === live.id);
        if (idx >= 0) {
          // 列表已有同 id（后端已落库该消息）：用流式对象替换，保证 flush/onEnd 引用一致
          list[idx] = live;
        } else {
          list.push(live);
        }
      }
      messages.value = list;
      if (scope.value === "self") {
        markConversationRead(conversationId);
      }
    } finally {
      messagesLoading.value = false;
    }
  }

  /**
   * 向上增量加载更早的历史消息（游标分页）。
   *
   * 以当前列表最早一条（正序首位）的 id 作为 `before`，仅取 id 更小的消息；
   * `hasMore` 直接取响应（不再靠"返回条数 < limit"推算）。已到底或并发重复调用短路。
   */
  async function loadMoreMessages() {
    const conversationId = currentConversationId.value;
    if (
      !conversationId ||
      messagesLoadingMore.value ||
      !messagesHasMore.value
    ) {
      return;
    }
    const earliest = messages.value[0];
    if (!earliest) return;
    messagesLoadingMore.value = true;
    try {
      const result = await AiConversationAPI.getMessages(conversationId, {
        before: earliest.id,
        limit: MESSAGES_PAGE_SIZE,
      });
      if (isStale(conversationId)) return;
      const descList = result.list ?? [];
      messagesHasMore.value = result.hasMore;
      if (descList.length === 0) return;
      collectThoughts(descList);
      // 该页 id 均小于当前最早消息，反转回时间正序后前置；按 id 去重防止边界重叠
      const existing = new Set(messages.value.map((m) => m.id));
      const older = descList
        .slice()
        .reverse()
        .filter((m) => !existing.has(m.id));
      if (older.length) messages.value = [...older, ...messages.value];
    } finally {
      messagesLoadingMore.value = false;
    }
  }

  async function markConversationRead(conversationId: number) {
    try {
      await AiConversationAPI.markConversationRead(conversationId);
      const conversation = conversations.value.find(
        (item) => item.id === conversationId
      );
      if (conversation) conversation.unreadCount = 0;
    } catch {
      // 已读标记失败不影响消息浏览
    }
  }

  async function deleteMessage(messageId: number) {
    await AiConversationAPI.deleteMessage(messageId);
    messages.value = messages.value.filter((item) => item.id !== messageId);
  }

  /**
   * 切换当前激活分支：调后端更新会话 currentBranchMessageId，成功后刷新消息列表
   * （复用 fetchMessages，不新增重复加载逻辑）。进行中防重入；失败原样抛出，
   * 由绑定层给用户可见提示，本地状态保持原样（不预置选中态）。
   */
  async function switchBranch(messageId: number) {
    const conversationId = currentConversationId.value;
    if (!conversationId || branchSwitching.value) return;
    branchSwitching.value = true;
    try {
      const conversation = await AiConversationAPI.switchBranch(
        conversationId,
        messageId
      );
      const local = conversations.value.find(
        (item) => item.id === conversationId
      );
      if (local) Object.assign(local, conversation);
      await fetchMessages(conversationId);
    } finally {
      branchSwitching.value = false;
    }
  }

  return {
    messagesLoading,
    messagesLoadingMore,
    messagesHasMore,
    branchSwitching,
    fetchMessages,
    loadMoreMessages,
    deleteMessage,
    markConversationRead,
    switchBranch,
  };
}
