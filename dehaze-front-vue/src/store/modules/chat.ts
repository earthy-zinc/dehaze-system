// AI 对话公共 Store：用户端（scope=self）与管理端会话审计（scope=admin）共用的消息数据视图层
// SSE 流式状态机：streaming → completed/failed/canceled，interrupt 时 paused，resume 续流
import {
  AiConversationAPI,
  type AiMessageVO,
  type ConversationCreateForm,
  type ConversationQuery,
  type ConversationStatus,
  type ConversationUpdateForm,
  type ConversationVO,
  type FeedbackForm,
  type FeedbackVO,
  type InterruptEvent,
  type MemoryVO,
  type MessageResumeForm,
  type MessageStreamHandlers,
  type ThoughtEvent,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { computed, reactive, ref } from "vue";
import { useVoiceStore } from "@/store/modules/voice";

export type ChatScope = "self" | "admin";
export type ConversationFilterStatus = ConversationStatus | 0;
export type BatchAction = "archive" | "restore" | "delete";

const MESSAGES_PAGE_SIZE = 50;
/** 流式超时：120 秒无新 token 判定失败 */
const STREAM_TIMEOUT_MS = 120_000;
const TIMEOUT_CHECK_INTERVAL_MS = 15_000;
/** 断线重连：最大 3 次，间隔 3 秒 */
const RECONNECT_MAX_ATTEMPTS = 3;
const RECONNECT_INTERVAL_MS = 3_000;
/** async_wait 挂起轮询：5 秒一次，最长 10 分钟 */
const ASYNC_POLL_INTERVAL_MS = 5_000;
const ASYNC_POLL_MAX_TIMES = 120;

interface ToolCallDraft {
  name?: string;
  args: string;
}

interface StreamSession {
  conversationId: number;
  controller: AbortController | null;
  /** assistant 消息当前 ID（新消息为占位负数，message.start 后替换） */
  messageId: number;
  /**
   * 流式消息对象引用（占位/当前消息）。流式增量（flush/onEnd）直接操作该引用，
   * 避免并发 fetchMessages 覆盖消息列表后 findMessage 找不到消息，
   * 导致内容不渲染、status 不清、永久"正在思考"。
   */
  message: AiMessageVO | null;
  textBuffer: string;
  /** 多段思考：每段思考一个字符串（后端按段推送 content_block.start/stop） */
  thinkingBlocks: string[];
  flushScheduled: boolean;
  lastTokenAt: number;
  streamSessionId?: string;
  /** 最近收到的 SSE 事件 ID（断点续传，per stream_session 计数，重连不可跨会话复用） */
  lastEventId?: string;
  reconnectAttempts: number;
  reconnectTimer: number | null;
  timeoutTimer: number | null;
  toolBlocks: Map<number, ToolCallDraft>;
  /** 流已到达终态（message.end / error / 用户停止），后续网络错误不再触发重连 */
  finished: boolean;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** 将 markdown 文本转为适合语音朗读的纯文本（手动朗读与自动播报共用，避免读出 `#`/`**` 等符号） */
function toSpeechText(markdown: string): string {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/[|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export const useChatStore = defineStore("chat", () => {
  const scope = ref<ChatScope>("self");

  // ===== 会话列表 =====
  const conversations = ref<ConversationVO[]>([]);
  const conversationsTotal = ref(0);
  const conversationsLoading = ref(false);
  const conversationQuery = reactive<ConversationQuery>({
    keyword: "",
    pageNum: 1,
    pageSize: 50,
  });

  // ===== 当前会话与消息 =====
  const currentConversationId = ref<number | null>(null);
  const messages = ref<AiMessageVO[]>([]);
  const messagesLoading = ref(false);

  // ===== 流式状态 =====
  const streamingMessageId = ref<number | null>(null);
  const interruptedMessageId = ref<number | null>(null);
  const interrupts = ref<InterruptEvent[]>([]);
  const suggestions = ref<string[]>([]);
  const thoughtsByMessage = ref<Record<number, ThoughtEvent[]>>({});
  const thinkingByMessage = ref<Record<number, string[]>>({});
  const toolCallsByMessage = ref<Record<number, unknown[]>>({});
  const messageMemories = ref<Record<number, MemoryVO[]>>({});

  // ===== 本地交互状态 =====
  const selectionMode = ref(false);
  const scrollFollowEnabled = ref(true);
  const quotedMessage = ref<AiMessageVO | null>(null);
  const feedbackByMessage = ref<Record<number, FeedbackVO | null>>({});

  const isStreaming = computed(
    () =>
      streamingMessageId.value !== null || interruptedMessageId.value !== null
  );

  let session: StreamSession | null = null;
  let asyncPollToken = 0;

  /** 注入数据范围，由宿主页面在挂载时调用 */
  function initScope(next: ChatScope) {
    if (scope.value === next) return;
    teardownStream();
    scope.value = next;
    conversations.value = [];
    messages.value = [];
    currentConversationId.value = null;
  }

  // ==================== 会话管理 ====================

  async function fetchConversations() {
    conversationsLoading.value = true;
    try {
      const query: ConversationQuery = { ...conversationQuery };
      if (scope.value === "admin") {
        query.view = "admin";
      }
      const result = await AiConversationAPI.getConversations(query);
      conversations.value = result.list ?? [];
      conversationsTotal.value = result.total ?? 0;
    } finally {
      conversationsLoading.value = false;
    }
  }

  async function createConversation(form?: ConversationCreateForm) {
    const conversation = await AiConversationAPI.createConversation(form);
    await fetchConversations();
    return conversation;
  }

  async function updateConversation(id: number, form: ConversationUpdateForm) {
    const conversation = await AiConversationAPI.updateConversation(id, form);
    const local = conversations.value.find((item) => item.id === id);
    if (local) Object.assign(local, conversation);
    return conversation;
  }

  async function deleteConversation(id: number) {
    await AiConversationAPI.deleteConversation(id);
    conversations.value = conversations.value.filter((item) => item.id !== id);
    if (currentConversationId.value === id) {
      teardownStream();
      currentConversationId.value = null;
      messages.value = [];
    }
  }

  async function batchOperateConversations(action: BatchAction, ids: number[]) {
    if (ids.length === 0) return;
    // 批量删除的二次确认由宿主页面负责，此处确认后携带 confirm 执行
    await AiConversationAPI.batchConversations({
      action,
      ids,
      confirm: action === "delete",
    });
    await fetchConversations();
    if (
      action === "delete" &&
      currentConversationId.value &&
      ids.includes(currentConversationId.value)
    ) {
      teardownStream();
      currentConversationId.value = null;
      messages.value = [];
    }
  }

  async function exportConversation(
    id: number,
    format: "json" | "markdown" = "markdown"
  ) {
    const blob = await AiConversationAPI.exportConversation(id, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `conversation-${id}.${format === "json" ? "json" : "md"}`;
    link.click();
    URL.revokeObjectURL(url);
  }

  // ==================== 消息 ====================

  async function fetchMessages(conversationId: number) {
    if (session && session.conversationId !== conversationId) {
      teardownStream();
    }
    currentConversationId.value = conversationId;
    messagesLoading.value = true;
    try {
      const result = await AiConversationAPI.getMessages(conversationId, {
        pageNum: 1,
        pageSize: MESSAGES_PAGE_SIZE,
      });
      const descList = result.list ?? [];
      // 后端分页为倒序（pageNum=1 返回最新一页），反转回时间正序展示（最早在上）
      const list = descList.slice().reverse();
      // 历史思考链写入（列表接口已按 assistant 消息附带 thoughts）
      for (const m of descList) {
        if (m.role === "assistant" && m.thoughts?.length) {
          thoughtsByMessage.value[m.id] = m.thoughts;
        }
      }
      // 流式进行中（同一会话）：保留当前流式消息对象，避免整体覆盖后
      // findMessage 找不到消息（content 不渲染、status 不清，界面永久"正在思考"）
      const live =
        session?.conversationId === conversationId ? session.message : null;
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

  function findMessage(messageId: number) {
    return messages.value.find((item) => item.id === messageId);
  }

  // ==================== SSE 流式状态机 ====================

  function scheduleFlush(s: StreamSession) {
    if (s.flushScheduled) return;
    s.flushScheduled = true;
    requestAnimationFrame(() => {
      s.flushScheduled = false;
      const msg = s.message ?? findMessage(s.messageId);
      if (!msg) return;
      if (s.textBuffer) {
        // 防御：流式正文只允许写入 assistant 消息，杜绝任何情况下拼到用户消息
        if (msg.role === "assistant") {
          msg.content = (msg.content ?? "") + s.textBuffer;
        }
        s.textBuffer = "";
      }
      if (s.thinkingBlocks.length) {
        const list = thinkingByMessage.value[s.messageId] ?? [];
        thinkingByMessage.value[s.messageId] = list.concat(s.thinkingBlocks);
        s.thinkingBlocks = [];
      }
    });
  }

  function appendThought(messageId: number, thought: ThoughtEvent) {
    const list =
      thoughtsByMessage.value[messageId] ??
      (thoughtsByMessage.value[messageId] = []);
    const index = list.findIndex((item) => item.position === thought.position);
    if (index >= 0) list[index] = thought;
    else list.push(thought);
    list.sort((a, b) => a.position - b.position);
  }

  function finishSession(s: StreamSession) {
    s.finished = true;
    if (s.reconnectTimer !== null) window.clearTimeout(s.reconnectTimer);
    if (s.timeoutTimer !== null) window.clearInterval(s.timeoutTimer);
    if (session === s) session = null;
  }

  function failMessage(s: StreamSession, reason: string) {
    const msg = s.message ?? findMessage(s.messageId);
    if (msg) {
      msg.status = 3;
      msg.error = reason;
    }
    streamingMessageId.value = null;
    finishSession(s);
  }

  /** async_wait 中断：message.end 仅表示通道关闭，消息未完成，轮询直到 status 到终态 */
  async function pollAsyncMessage(conversationId: number, messageId: number) {
    const token = ++asyncPollToken;
    for (let i = 0; i < ASYNC_POLL_MAX_TIMES; i++) {
      await sleep(ASYNC_POLL_INTERVAL_MS);
      if (
        token !== asyncPollToken ||
        currentConversationId.value !== conversationId
      )
        return;
      try {
        const detail = await AiConversationAPI.getMessageDetail(messageId);
        if (detail.status >= 2) {
          const msg = findMessage(messageId);
          if (msg) Object.assign(msg, detail);
          interrupts.value = interrupts.value.filter(
            (item) => item.type !== "async_wait"
          );
          interruptedMessageId.value = null;
          return;
        }
      } catch {
        // 单次轮询失败忽略，下一轮重试
      }
    }
  }

  /** 语音回复开关开启时自动朗读助手回复（仅用户端；合成失败静默降级，不打断对话主流程） */
  function autoSpeak(message: AiMessageVO) {
    if (scope.value !== "self") return;
    const voiceStore = useVoiceStore();
    if (!voiceStore.ttsPreference.enabled) return;
    const text = toSpeechText(message.content ?? "");
    if (!text) return;
    void voiceStore.playSpeech(text).catch(() => {
      // 降级为纯文本回复（需求规格 §3.2.4）
    });
  }

  function buildHandlers(s: StreamSession): MessageStreamHandlers {
    return {
      onStart(data) {
        let msg: AiMessageVO | null =
          s.message ?? findMessage(s.messageId) ?? null;
        if (!msg) {
          // 兜底：占位可能已被并发消息加载覆盖，回退到流式中的 assistant 消息
          msg =
            messages.value.find(
              (item) => item.role === "assistant" && item.status === 1
            ) ?? null;
        }
        s.messageId = data.messageId;
        if (msg) {
          msg.id = data.messageId;
          s.message = msg;
        }
        streamingMessageId.value = data.messageId;
        if (data.streamSessionId) s.streamSessionId = data.streamSessionId;
        s.lastTokenAt = Date.now();
      },
      onContentBlockStart(data) {
        s.lastTokenAt = Date.now();
        if (data.type === "tool_use") {
          s.toolBlocks.set(data.index, { args: "" });
        } else if (data.type === "thinking") {
          // 多段思考：每段思考一个独立内容块，逐段累积
          s.thinkingBlocks.push("");
        }
      },
      onContentBlockDelta(data) {
        s.lastTokenAt = Date.now();
        if (data.delta.type === "text_delta") {
          s.textBuffer += data.delta.text ?? "";
        } else if (data.delta.type === "thinking_delta") {
          // 兼容无 start 的残留流
          if (s.thinkingBlocks.length === 0) s.thinkingBlocks.push("");
          s.thinkingBlocks[s.thinkingBlocks.length - 1] +=
            data.delta.thinking ?? "";
        } else if (data.delta.type === "input_json_delta") {
          const block = s.toolBlocks.get(data.index);
          if (block) {
            if (data.delta.name) block.name = data.delta.name;
            block.args += data.delta.partialJson ?? "";
          }
        }
        scheduleFlush(s);
      },
      onContentBlockStop(data) {
        const block = s.toolBlocks.get(data.index);
        if (!block) return;
        s.toolBlocks.delete(data.index);
        let parsed: unknown = block.args;
        try {
          parsed = JSON.parse(block.args);
        } catch {
          // 非法 JSON 保留原始字符串
        }
        const list =
          toolCallsByMessage.value[s.messageId] ??
          (toolCallsByMessage.value[s.messageId] = []);
        list.push({ name: block.name, arguments: parsed });
      },
      onThought(data) {
        s.lastTokenAt = Date.now();
        appendThought(s.messageId, data);
      },
      onInterrupt(data) {
        interrupts.value.push(data);
        interruptedMessageId.value = s.messageId;
        streamingMessageId.value = null;
        // 流会随 message.end 关闭，标记完成等待 onEnd 收尾
        finishSession(s);
      },
      onSuggestions(data) {
        suggestions.value = data.questions.map((item) => item.question);
      },
      onEventId(id) {
        s.lastEventId = id;
      },
      onPing() {
        // 心跳仅保活，不重置 token 超时计时
      },
      onError(data) {
        failMessage(s, data.message || "推理服务返回错误");
      },
      onEnd(data) {
        const msg = s.message ?? findMessage(s.messageId);
        if (msg) {
          if (data.usage) {
            msg.inputTokens = data.usage.inputTokens;
            msg.outputTokens = data.usage.outputTokens;
            msg.cachedInputTokens = data.usage.cachedInputTokens;
            msg.credits = data.usage.credits;
          }
          if (interruptedMessageId.value === s.messageId) {
            // 中断挂起：保持生成中状态，等待 resume 或 async_wait 轮询
            if (interrupts.value.some((item) => item.type === "async_wait")) {
              void pollAsyncMessage(s.conversationId, s.messageId);
            }
          } else if (data.stopReason === "canceled") {
            msg.status = 4;
          } else if (
            data.stopReason === "error" ||
            data.stopReason === "content_filter"
          ) {
            msg.status = 3;
          } else {
            msg.status = 2;
            autoSpeak(msg);
          }
          const toolCalls = toolCallsByMessage.value[s.messageId];
          if (toolCalls?.length) msg.toolCalls = toolCalls;
        }
        streamingMessageId.value = null;
        finishSession(s);
      },
      onNetworkError() {
        handleStreamDisconnect(s);
      },
      onClose() {
        if (!s.finished) {
          handleStreamDisconnect(s);
        }
      },
    };
  }

  /** 网络断开：有流式会话 ID 时自动重连（最大 3 次间隔 3 秒），否则判定失败 */
  function handleStreamDisconnect(s: StreamSession) {
    if (s.finished) return;
    if (s.streamSessionId && s.reconnectAttempts < RECONNECT_MAX_ATTEMPTS) {
      s.reconnectAttempts++;
      s.reconnectTimer = window.setTimeout(() => {
        AiConversationAPI.reconnectStream(
          s.conversationId,
          s.streamSessionId!,
          s.lastEventId ?? "",
          buildHandlers(s)
        );
      }, RECONNECT_INTERVAL_MS);
      return;
    }
    failMessage(s, "网络连接中断，请检查网络后重试");
  }

  function startTimeoutWatch(s: StreamSession) {
    s.lastTokenAt = Date.now();
    s.timeoutTimer = window.setInterval(() => {
      if (Date.now() - s.lastTokenAt > STREAM_TIMEOUT_MS) {
        s.controller?.abort();
        failMessage(s, "流式输出超时（120 秒无新内容）");
      }
    }, TIMEOUT_CHECK_INTERVAL_MS);
  }

  /**
   * 打开流式会话。opener 负责调用 SDK 建立 SSE 连接（sendMessage/regenerate/edit/resume/reconnect）。
   * 同一会话同一时间仅允许一个流式输出进行中。
   */
  function openStream(
    conversationId: number,
    messageId: number,
    opener: (handlers: MessageStreamHandlers) => AbortController | void
  ) {
    if (session) {
      ElMessage.warning("当前会话正在推理中，请等待完成或先停止");
      return false;
    }
    const s: StreamSession = {
      conversationId,
      controller: null,
      messageId,
      message: findMessage(messageId) ?? null,
      textBuffer: "",
      thinkingBlocks: [],
      flushScheduled: false,
      lastTokenAt: Date.now(),
      reconnectAttempts: 0,
      reconnectTimer: null,
      timeoutTimer: null,
      toolBlocks: new Map(),
      finished: false,
    };
    s.controller = opener(buildHandlers(s)) ?? null;
    session = s;
    startTimeoutWatch(s);
    return true;
  }

  function appendAssistantPlaceholder(
    conversationId: number,
    parentMessageId?: number
  ) {
    const placeholder: AiMessageVO = {
      id: -Date.now(),
      conversationId,
      role: "assistant",
      content: "",
      status: 1,
      parentMessageId,
      createTime: new Date().toISOString(),
    };
    messages.value.push(placeholder);
    return placeholder;
  }

  // ==================== 消息操作 ====================

  function sendMessage(content: string, model?: string) {
    const conversationId = currentConversationId.value;
    if (!conversationId || !content.trim()) return;
    // 引用预填充内容合并为消息内容后发送
    let finalContent = content;
    if (quotedMessage.value) {
      finalContent = `> ${quotedMessage.value.content}\n\n${content}`;
      quotedMessage.value = null;
    }
    const userMessage: AiMessageVO = {
      id: -Date.now() - 1,
      conversationId,
      role: "user",
      content: finalContent,
      status: 2,
      createTime: new Date().toISOString(),
    };
    messages.value.push(userMessage);
    const placeholder = appendAssistantPlaceholder(
      conversationId,
      userMessage.id
    );
    interrupts.value = [];
    suggestions.value = [];
    interruptedMessageId.value = null;
    streamingMessageId.value = placeholder.id;
    openStream(conversationId, placeholder.id, (handlers) =>
      AiConversationAPI.sendMessage(
        conversationId,
        { content: finalContent, model },
        handlers
      )
    );
  }

  function regenerate(messageId: number) {
    const conversationId = currentConversationId.value;
    if (!conversationId) return;
    const original = findMessage(messageId);
    const placeholder = appendAssistantPlaceholder(
      conversationId,
      original?.parentMessageId ?? original?.id
    );
    interrupts.value = [];
    suggestions.value = [];
    interruptedMessageId.value = null;
    streamingMessageId.value = placeholder.id;
    openStream(conversationId, placeholder.id, (handlers) =>
      AiConversationAPI.regenerate(messageId, handlers)
    );
  }

  function editMessage(messageId: number, content: string) {
    const conversationId = currentConversationId.value;
    if (!conversationId || !content.trim()) return;
    const placeholder = appendAssistantPlaceholder(conversationId, messageId);
    interrupts.value = [];
    suggestions.value = [];
    interruptedMessageId.value = null;
    streamingMessageId.value = placeholder.id;
    openStream(conversationId, placeholder.id, (handlers) =>
      AiConversationAPI.editMessage(messageId, { content }, handlers)
    );
  }

  function resumeInterrupt(messageId: number, data: MessageResumeForm = {}) {
    const conversationId = currentConversationId.value;
    if (!conversationId) return;
    interrupts.value = [];
    interruptedMessageId.value = null;
    const msg = findMessage(messageId);
    if (msg) msg.status = 1;
    streamingMessageId.value = messageId;
    openStream(conversationId, messageId, (handlers) =>
      AiConversationAPI.resumeMessage(messageId, data, handlers)
    );
  }

  /** 断线重连：从当前会话中重新挂载仍处于生成中的 assistant 消息 */
  function reconnectStream(streamSessionId: string, lastEventId = "") {
    const conversationId = currentConversationId.value;
    if (!conversationId) return;
    const msg = messages.value.find(
      (item) => item.role === "assistant" && item.status === 1
    );
    if (!msg) return;
    openStream(conversationId, msg.id, (handlers) => {
      AiConversationAPI.reconnectStream(
        conversationId,
        streamSessionId,
        lastEventId,
        handlers
      );
    });
  }

  async function stopStreaming() {
    const messageId = streamingMessageId.value ?? interruptedMessageId.value;
    if (!messageId) return;
    session?.controller?.abort();
    finishSessionPartial();
    const msg = findMessage(messageId);
    if (msg) msg.status = 4;
    streamingMessageId.value = null;
    interruptedMessageId.value = null;
    // 占位消息（负数 ID）尚未在服务端落库，无需调用 stop 接口
    if (messageId > 0) {
      try {
        const result = await AiConversationAPI.stopMessage(messageId);
        if (msg) Object.assign(msg, result);
      } catch {
        // 停止接口失败不影响本地取消态
      }
    }
  }

  function finishSessionPartial() {
    if (session) finishSession(session);
    asyncPollToken++;
  }

  async function submitFeedback(messageId: number, data: FeedbackForm | null) {
    if (data) {
      const feedback = await AiConversationAPI.submitFeedback(messageId, data);
      feedbackByMessage.value[messageId] = feedback;
    } else {
      await AiConversationAPI.deleteFeedback(messageId);
      feedbackByMessage.value[messageId] = null;
    }
  }

  async function fetchFeedback(messageId: number) {
    if (messageId <= 0) return;
    try {
      feedbackByMessage.value[messageId] =
        (await AiConversationAPI.getFeedback(messageId)) ?? null;
    } catch {
      // 反馈查询失败按无反馈展示
    }
  }

  async function deleteMessage(messageId: number) {
    await AiConversationAPI.deleteMessage(messageId);
    messages.value = messages.value.filter((item) => item.id !== messageId);
  }

  function quoteMessage(message: AiMessageVO) {
    quotedMessage.value = message;
  }

  async function speakMessage(message: AiMessageVO) {
    const voiceStore = useVoiceStore();
    if (voiceStore.playbackState === "playing") {
      voiceStore.stopSpeech();
      return;
    }
    // 空文本（工具调用消息/纯参数消息）无朗读语义，直接返回，避免 TTS 空文本参数错误
    const text = toSpeechText(message.content ?? "");
    if (!text) {
      return;
    }
    await voiceStore.playSpeech(text);
  }

  function applySuggestion(question: string) {
    suggestions.value = [];
    sendMessage(question);
  }

  function teardownStream() {
    session?.controller?.abort();
    finishSessionPartial();
    streamingMessageId.value = null;
    interruptedMessageId.value = null;
  }

  return {
    scope,
    conversations,
    conversationsTotal,
    conversationsLoading,
    conversationQuery,
    currentConversationId,
    messages,
    messagesLoading,
    streamingMessageId,
    interruptedMessageId,
    interrupts,
    suggestions,
    thoughtsByMessage,
    thinkingByMessage,
    toolCallsByMessage,
    messageMemories,
    selectionMode,
    scrollFollowEnabled,
    quotedMessage,
    feedbackByMessage,
    isStreaming,
    initScope,
    fetchConversations,
    createConversation,
    updateConversation,
    deleteConversation,
    batchOperateConversations,
    exportConversation,
    fetchMessages,
    markConversationRead,
    sendMessage,
    regenerate,
    editMessage,
    resumeInterrupt,
    reconnectStream,
    stopStreaming,
    submitFeedback,
    fetchFeedback,
    deleteMessage,
    quoteMessage,
    speakMessage,
    applySuggestion,
    teardownStream,
  };
});
