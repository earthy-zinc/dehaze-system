// 流式 slice：公开动作编排与流会话生命周期持有。
//
// 逐事件归约在 stream-reducer，事件分派在 stream-events，副作用在 stream-effects，
// 连接/重连/超时在 stream-transport；本文件只做动作串联与全局流状态归属。
import {
  AiConversationAPI,
  type MessageResume,
  type MessageStreamHandlers,
} from "clarity-sdk-js";
import { textOfBlocks } from "@/components/ai-chat/model/blocks";
import { useVoiceStore } from "@/store/modules/voice";
import {
  createLocalMessage,
  findMessage,
  putMessage,
  removeMessage,
} from "./entities";
import { markConversationRead } from "./messages";
import {
  toSpeechText,
  type ChatCtx,
  type ChatScope,
  type StreamSession,
  type StreamStatus,
} from "./shared";
import { createStreamEffects } from "./stream-effects";
import { createStreamEvents } from "./stream-events";
import { buildResumeForm } from "./stream-resume";
import {
  abortStream,
  createStreamSession,
  handleStreamDisconnect,
  startStream,
  type StreamTransportHooks,
} from "./stream-transport";

export function createStreamSlice(ctx: ChatCtx) {
  const {
    scope,
    currentConversationId,
    streamStatus,
    streamMessageId,
    streamSessionId,
    quotedMessage,
  } = ctx;

  /** 写全局流式态（唯一的三项：status / messageId / sessionId） */
  function setStream(status: StreamStatus, messageId: number | null) {
    streamStatus.value = status;
    streamMessageId.value = messageId;
    if (status !== "running") streamSessionId.value = null;
  }

  /** 流终态收尾：清定时器、释放会话；当前会话在流终态补一次已读 */
  function finishSession(s: StreamSession) {
    if (s.finished) return;
    s.finished = true;
    if (s.reconnectTimer !== null) window.clearTimeout(s.reconnectTimer);
    if (s.timeoutTimer !== null) window.clearInterval(s.timeoutTimer);
    if (ctx.session === s) ctx.session = null;
    streamSessionId.value = null;
    if (
      s.originScope === "self" &&
      currentConversationId.value === s.conversationId
    ) {
      void markConversationRead(ctx, s.conversationId);
    }
  }

  const effects = createStreamEffects(ctx, { setStream, finishSession });

  const events = createStreamEvents(ctx, {
    applyEffect: effects.applyEffect,
    fail: effects.failMessage,
    onDisconnect: (session) => handleStreamDisconnect(session, transportHooks),
  });

  const transportHooks: StreamTransportHooks = {
    buildHandlers: events.buildHandlers,
    fail: effects.failMessage,
    onInterruptedClose: effects.settleInterruptedClose,
  };

  /** 同一时间仅允许一个流：忙时统一提示（调用方据此短路） */
  function isBusy(): boolean {
    if (!ctx.session) return false;
    ElMessage.warning("当前会话正在推理中，请等待完成或先停止");
    return true;
  }

  /**
   * 打开流会话。opener 负责调用 SDK 建立 SSE 连接。
   * 返回 false 表示未开启（会话忙），调用方据此回滚本地占位。
   */
  function openStream(
    conversationId: number,
    messageId: number,
    localUserMessageId: number | null,
    opener: (handlers: MessageStreamHandlers) => AbortController | void,
    options: { clearInterruptOnEvent?: boolean; originScope?: ChatScope } = {}
  ): boolean {
    if (isBusy()) return false;
    const session = createStreamSession({
      conversationId,
      messageId,
      localUserMessageId,
      originScope: options.originScope ?? scope.value,
      clearInterruptOnEvent: options.clearInterruptOnEvent ?? false,
    });
    ctx.session = session;
    setStream("running", messageId);
    startStream(session, opener, transportHooks);
    return true;
  }

  /**
   * 本地起一轮：写可选用户消息与助手占位后开流；未开启（会话忙）时回滚本地实体。
   * 不带 userText 时仅创建助手占位（重新生成），其父消息由调用方指定。
   */
  function runTurn(
    conversationId: number,
    params: {
      userText?: string;
      userParentMessageId?: number | null;
      assistantParentMessageId?: number | null;
    },
    opener: (handlers: MessageStreamHandlers) => AbortController | void
  ): void {
    const userMessage = params.userText
      ? createLocalMessage({
          conversationId,
          role: "user",
          status: "completed",
          text: params.userText,
          parentMessageId: params.userParentMessageId ?? null,
        })
      : null;
    if (userMessage) putMessage(ctx, userMessage);
    const placeholder = createLocalMessage({
      conversationId,
      role: "assistant",
      status: "running",
      parentMessageId:
        userMessage?.id ?? params.assistantParentMessageId ?? null,
    });
    putMessage(ctx, placeholder);
    const started = openStream(
      conversationId,
      placeholder.id,
      userMessage?.id ?? null,
      opener
    );
    if (!started) {
      if (userMessage) removeMessage(ctx, userMessage.id);
      removeMessage(ctx, placeholder.id);
      setStream("idle", null);
    }
  }

  function sendMessage(content: string, model?: string) {
    const conversationId = currentConversationId.value;
    if (!conversationId || !content.trim() || isBusy()) return;
    // 引用预填充内容合并为消息内容后发送
    let finalContent = content;
    const quoted = quotedMessage.value;
    if (quoted) {
      finalContent = `> ${textOfBlocks(quoted.blocks)}\n\n${content}`;
      quotedMessage.value = null;
    }
    runTurn(conversationId, { userText: finalContent }, (handlers) =>
      AiConversationAPI.sendMessage(
        conversationId,
        { content: finalContent, model },
        handlers
      )
    );
  }

  function regenerate(messageId: number) {
    const conversationId = currentConversationId.value;
    const original = findMessage(ctx, messageId);
    if (!conversationId || !original || isBusy()) return;
    runTurn(
      conversationId,
      { assistantParentMessageId: original.parentMessageId ?? original.id },
      (handlers) => AiConversationAPI.regenerate(messageId, handlers)
    );
  }

  function editMessage(messageId: number, content: string) {
    const conversationId = currentConversationId.value;
    const original = findMessage(ctx, messageId);
    if (!conversationId || !original || !content.trim() || isBusy()) return;
    original.edited = true;
    runTurn(
      conversationId,
      {
        userText: content,
        userParentMessageId: original.parentMessageId ?? null,
      },
      (handlers) =>
        AiConversationAPI.editMessage(messageId, { content }, handlers)
    );
  }

  function resumeInterrupt(messageId: number, data: MessageResume = {}) {
    const conversationId = currentConversationId.value;
    const message = findMessage(ctx, messageId);
    if (!conversationId || !message) return;
    const pending = message.pendingInterrupt ?? undefined;
    const form = buildResumeForm(pending, data);
    if (form === null) return;
    const previousStatus = message.status;
    message.status = "running";
    message.error = null;
    const started = openStream(
      conversationId,
      messageId,
      null,
      (handlers) => AiConversationAPI.resumeMessage(messageId, form, handlers),
      { clearInterruptOnEvent: true }
    );
    if (!started) {
      message.status = previousStatus;
      setStream(pending ? "interrupted" : "idle", pending ? messageId : null);
    }
  }

  async function stopStreaming(targetId?: number) {
    const messageId = targetId ?? streamMessageId.value;
    if (messageId === null) return;
    abortStream(ctx.session);
    if (ctx.session) finishSession(ctx.session);
    effects.cancelAsyncPoll();
    setStream("idle", null);
    const message = findMessage(ctx, messageId);
    if (
      message &&
      (message.status === "running" || message.status === "awaiting_input")
    ) {
      message.status = "canceled";
      message.pendingInterrupt = null;
    }
    // 占位消息（负数 ID）尚未在服务端落库，无需调用 stop 接口
    if (messageId > 0) {
      try {
        const result = await AiConversationAPI.stopMessage(messageId);
        putMessage(ctx, result);
      } catch {
        // 停止接口失败不影响本地取消态
      }
    }
  }

  function quoteMessage(messageId: number) {
    quotedMessage.value = findMessage(ctx, messageId) ?? null;
  }

  async function speakMessage(messageId: number) {
    const message = findMessage(ctx, messageId);
    if (!message) return;
    const voiceStore = useVoiceStore();
    if (voiceStore.playbackState === "playing") {
      voiceStore.stopSpeech();
      return;
    }
    const text = toSpeechText(textOfBlocks(message.blocks));
    if (!text) return;
    await voiceStore.playSpeech(text);
  }

  function applySuggestion(question: string) {
    sendMessage(question);
  }

  /** 离开会话/切换 scope：断开流并复位全局流式态（实体保留，历史按需重载） */
  function teardownStream() {
    abortStream(ctx.session);
    if (ctx.session) finishSession(ctx.session);
    effects.cancelAsyncPoll();
    setStream("idle", null);
  }

  return {
    sendMessage,
    regenerate,
    editMessage,
    resumeInterrupt,
    stopStreaming,
    quoteMessage,
    speakMessage,
    applySuggestion,
    teardownStream,
  };
}
