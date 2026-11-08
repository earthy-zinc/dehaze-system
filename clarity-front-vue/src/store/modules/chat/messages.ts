// 消息 slice：历史消息加载（游标分页）、消息级操作与实体表刷新
import { AiConversationAPI, type Message } from "clarity-sdk-js";
import { ref } from "vue";
import { MESSAGES_PAGE_SIZE, type ChatCtx } from "./shared";
import { removeMessage } from "./entities";

/**
 * 历史消息加载错误：initial 指首屏/整表刷新失败，more 指上翻更早历史失败。
 * 失败不清空已有消息、不改 hasMore——列表只负责提示与重试，不猜测"到底"。
 */
export interface MessagesError {
  kind: "initial" | "more";
  message: string;
}

/** 失败原因取接口/运行时给出的真实消息；异常值无文案时给中性提示，不编造业务原因 */
function errorMessageOf(error: unknown): string {
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : "请求失败";
}

/**
 * 标记会话已读：调后端更新 last_read_message_id，并把本地列表未读清零。
 * 失败不影响消息浏览（仅未读红点保留）。
 */
export async function markConversationRead(
  ctx: ChatCtx,
  conversationId: number
): Promise<void> {
  try {
    await AiConversationAPI.markConversationRead(conversationId);
    const conversation = ctx.conversations.value.find(
      (item) => item.id === conversationId
    );
    if (conversation) conversation.unreadCount = 0;
  } catch {
    // 已读标记失败不影响消息浏览
  }
}

export function createMessagesSlice(ctx: ChatCtx, teardownStream: () => void) {
  const {
    messages,
    messageOrder,
    currentConversationId,
    scope,
    conversations,
  } = ctx;

  const messagesLoading = ref(false);
  const messagesLoadingMore = ref(false);
  const messagesHasMore = ref(false);
  /** 历史加载失败态：列表据此给出行内错误条与重试入口（成功后清空） */
  const messagesError = ref<MessagesError | null>(null);
  /** 分支切换进行中：禁用切换器并防重复提交 */
  const branchSwitching = ref(false);

  /** 分页请求期间切换会话时丢弃过期结果（避免把 A 会话的历史拼到 B 会话） */
  function isStale(conversationId: number) {
    return currentConversationId.value !== conversationId;
  }

  /** 当前会话是否有进行中的流（同会话刷新需保留流式实体引用） */
  function liveSession() {
    return ctx.session &&
      ctx.session.conversationId === currentConversationId.value
      ? ctx.session
      : null;
  }

  /**
   * 用历史页整体替换实体表。
   *
   * - 同会话进行中的流：流式实体（含未绑定的本地占位）原样保留，避免覆盖后事件写不进去；
   * - 刷新后的 running 僵尸：没有活跃流承接的 running 消息不可能再被推进，按 failed 收口，
   *   否则界面永久停在"正在思考"。
   */
  function replaceAll(list: readonly Message[]) {
    const record: Record<number, Message> = {};
    const order: number[] = [];
    for (const entity of list) {
      record[entity.id] = entity;
      order.push(entity.id);
    }

    const session = liveSession();
    if (session) {
      for (const id of [session.localUserMessageId, session.messageId]) {
        if (id == null) continue;
        const live = messages.value[id];
        if (!live) continue;
        if (record[id]) {
          record[id] = live;
        } else {
          record[id] = live;
          order.push(id);
        }
      }
    }

    const activeId = session ? session.messageId : null;
    for (const entity of Object.values(record)) {
      if (entity.status === "running" && entity.id !== activeId) {
        entity.status = "failed";
        entity.error = "上次回复未完成";
      }
    }

    messages.value = record;
    messageOrder.value = order;
    // 实体整体换新：版本号清零，投影全量重算一次
  }

  async function fetchMessages(conversationId: number) {
    // 切到另一个会话：结束上一个会话的流（含无活跃 session 的 interrupted 态），
    // 并在发请求前清空旧会话内容——否则请求失败后 A 的消息会留在 B 会话下。
    if (
      currentConversationId.value !== null &&
      currentConversationId.value !== conversationId
    ) {
      teardownStream();
      messages.value = {};
      messageOrder.value = [];
      messagesHasMore.value = false;
      // 旧会话的在途分页请求已作废（其结果按 isStale 丢弃且不再回落本标志）
      messagesLoadingMore.value = false;
    }
    currentConversationId.value = conversationId;
    messagesLoading.value = true;
    messagesError.value = null;
    try {
      // 首屏：缺省 before 取最新一页（游标分页按 id 倒序返回）
      const result = await AiConversationAPI.getMessages(conversationId, {
        limit: MESSAGES_PAGE_SIZE,
      });
      if (isStale(conversationId)) return;
      const descList = result.list ?? [];
      // 后端游标分页按 id 倒序返回，反转回时间正序展示（最早在上）
      replaceAll(descList.slice().reverse());
      messagesHasMore.value = result.hasMore;
      if (scope.value === "self") {
        void markConversationRead(ctx, conversationId);
      }
    } catch (error) {
      // 过期请求的失败属于上一个会话，不能记到当前会话头上
      if (!isStale(conversationId)) {
        messagesError.value = {
          kind: "initial",
          message: errorMessageOf(error),
        };
      }
      // 调用方（首屏/建会话后发送）需要知道加载失败，由页面决定是否继续
      throw error;
    } finally {
      if (!isStale(conversationId)) messagesLoading.value = false;
    }
  }

  /**
   * 向上增量加载更早的历史消息（游标分页）。
   *
   * 以当前列表最早一条正数 id 作为 `before`（本地占位为负数，不能作游标），
   * 仅取 id 更小的消息；`hasMore` 直接取响应。已到底或并发重复调用短路。
   * 失败只记录错误并保留原 hasMore：不能把"上翻失败"误判成"没有更多"。
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
    const earliest = messageOrder.value.find((id) => id > 0);
    if (earliest === undefined) return;
    messagesLoadingMore.value = true;
    messagesError.value = null;
    try {
      const result = await AiConversationAPI.getMessages(conversationId, {
        before: earliest,
        limit: MESSAGES_PAGE_SIZE,
      });
      if (isStale(conversationId)) return;
      const descList = result.list ?? [];
      messagesHasMore.value = result.hasMore;
      if (descList.length === 0) return;
      const existing = new Set(messageOrder.value);
      const older = descList
        .slice()
        .reverse()
        .filter((message) => !existing.has(message.id));
      if (older.length === 0) return;
      const record = { ...messages.value };
      for (const message of older) {
        record[message.id] = message;
      }
      messages.value = record;
      messageOrder.value = [
        ...older.map((message) => message.id),
        ...messageOrder.value,
      ];
    } catch (error) {
      if (!isStale(conversationId)) {
        messagesError.value = { kind: "more", message: errorMessageOf(error) };
      }
      // 上翻由滚动/按钮触发，没有等待结果的调用方：不 rethrow，避免未处理拒绝
    } finally {
      if (!isStale(conversationId)) messagesLoadingMore.value = false;
    }
  }

  /**
   * 重试历史加载：上翻失败重试同一游标，否则重新拉取当前会话。
   * fetch 的 rethrow 在此收口——错误已进入 messagesError，由列表展示。
   */
  async function retryMessages() {
    if (messagesError.value?.kind === "more") {
      await loadMoreMessages();
      return;
    }
    const conversationId = currentConversationId.value;
    if (!conversationId) return;
    try {
      await fetchMessages(conversationId);
    } catch {
      // 失败已记录到 messagesError
    }
  }

  async function deleteMessage(messageId: number) {
    await AiConversationAPI.deleteMessage(messageId);
    removeMessage(ctx, messageId);
    // 删掉的正是当前流/中断挂起的消息：复位全局流式态，避免指向已不存在的实体
    if (ctx.streamMessageId.value === messageId) teardownStream();
  }

  /**
   * 切换当前激活分支：调后端更新会话 currentBranchMessageId，成功后刷新消息列表
   * （复用 fetchMessages，不新增重复加载逻辑）。进行中防重入。
   * 指针接口失败向上抛（切换没发生）；刷新失败只进 messagesError——
   * 指针已切换，不能把"列表刷新失败"误报成"切换分支失败"。
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
      try {
        await fetchMessages(conversationId);
      } catch {
        // 错误已记录为 messagesError（列表展示行内重试）
      }
    } finally {
      branchSwitching.value = false;
    }
  }

  return {
    messagesLoading,
    messagesLoadingMore,
    messagesHasMore,
    messagesError,
    branchSwitching,
    fetchMessages,
    loadMoreMessages,
    retryMessages,
    deleteMessage,
    markConversationRead: (conversationId: number) =>
      markConversationRead(ctx, conversationId),
    switchBranch,
  };
}
