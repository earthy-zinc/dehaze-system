// 流式 slice：SSE 状态机（streaming → completed/failed/canceled，interrupt 时 paused，resume 续流）。
// 逐事件的状态迁移收敛为纯归约器 stream-reducer（reduceStreamState）；本文件只负责把归约产出的
// effects 解释执行为副作用（写 ref、rAF 批量写回、发请求、终态收尾），不在此直接做视图状态迁移。
import {
  AiConversationAPI,
  type AiMessageVO,
  type InterruptEvent,
  type MessageResumeForm,
  type MessageStreamHandlers,
} from "dehaze-sdk-js";
import {
  appendThinkingDelta,
  finalizeThinking as finalizeThinkingVM,
} from "@/components/ai-chat/vm/thinking";
import { useVoiceStore } from "@/store/modules/voice";
import {
  ASYNC_POLL_INTERVAL_MS,
  ASYNC_POLL_MAX_TIMES,
  RECONNECT_INTERVAL_MS,
  RECONNECT_MAX_ATTEMPTS,
  STREAM_TIMEOUT_MS,
  TIMEOUT_CHECK_INTERVAL_MS,
  nextLocalMessageId,
  resolveConfirmKind,
  sleep,
  toSpeechText,
  type ChatCtx,
  type StreamSession,
} from "./shared";
import {
  reduceStreamState,
  type StreamEffect,
  type StreamEvent,
} from "./stream-reducer";
import {
  appendAssistantPlaceholder as appendAssistantPlaceholderIn,
  findMessage as findMessageIn,
} from "./messages";

export function createStreamSlice(ctx: ChatCtx) {
  const {
    scope,
    messages,
    currentConversationId,
    streamingMessageId,
    interruptedMessageId,
    interrupts,
    suggestions,
    thoughtsByMessage,
    thinkingByMessage,
    toolCallsByMessage,
    planByMessage,
    subAgentsByMessage,
    quotedMessage,
  } = ctx;

  let asyncPollToken = 0;

  // 绑定到共享 ctx 的本地别名，保持内部调用点与原实现一致
  const findMessage = (messageId: number) => findMessageIn(ctx, messageId);
  const appendAssistantPlaceholder = (
    conversationId: number,
    parentMessageId?: number
  ) => appendAssistantPlaceholderIn(ctx, conversationId, parentMessageId);

  /**
   * rAF 批量写回：把归约缓冲（正文/思考增量）一次性并入消息与思考态，避免逐 delta 重渲染。
   * 逐 delta 事件只改纯归约状态（字符串缓冲），不触发任何响应式更新——性能语义与重构前一致。
   */
  function scheduleFlush(s: StreamSession) {
    if (s.flushScheduled) return;
    s.flushScheduled = true;
    requestAnimationFrame(() => {
      s.flushScheduled = false;
      const st = s.state;
      const msg = s.message ?? findMessage(st.messageId);
      if (!msg) return;
      let next = st;
      if (st.textBuffer) {
        // 防御：流式正文只允许写入 assistant 消息，杜绝任何情况下拼到用户消息
        if (msg.role === "assistant") {
          msg.content = (msg.content ?? "") + st.textBuffer;
        }
        next = { ...next, textBuffer: "" };
      }
      if (st.thinkingBuffer && st.thinking) {
        const thinking = appendThinkingDelta(st.thinking, st.thinkingBuffer);
        next = { ...next, thinking, thinkingBuffer: "" };
        thinkingByMessage.value[st.messageId] = thinking;
      }
      s.state = next;
    });
  }

  /** 流终态收尾：并入残留增量、闭合未关闭思考段并定格计时（重连不收尾，续流继续累积） */
  function finalizeThinking(s: StreamSession) {
    const st = s.state;
    if (!st.thinking) return;
    const base = st.thinkingBuffer
      ? appendThinkingDelta(st.thinking, st.thinkingBuffer)
      : st.thinking;
    const finalized = finalizeThinkingVM(base, Date.now());
    s.state = { ...st, thinking: finalized, thinkingBuffer: "" };
    thinkingByMessage.value[st.messageId] = finalized;
  }

  function finishSession(s: StreamSession) {
    finalizeThinking(s);
    s.finished = true;
    if (s.reconnectTimer !== null) window.clearTimeout(s.reconnectTimer);
    if (s.timeoutTimer !== null) window.clearInterval(s.timeoutTimer);
    if (ctx.session === s) ctx.session = null;
  }

  function failMessage(s: StreamSession, reason: string) {
    const msg = s.message ?? findMessage(s.state.messageId);
    if (msg) {
      msg.status = 3;
      msg.error = reason;
    }
    streamingMessageId.value = null;
    finishSession(s);
  }

  /** message.end 收尾：写用量/终态/工具调用；中断挂起时保持生成中并（async_wait）启动轮询 */
  function applyEnd(
    s: StreamSession,
    effect: Extract<StreamEffect, { type: "end" }>
  ) {
    const st = s.state;
    const msg = s.message ?? findMessage(st.messageId);
    if (msg) {
      if (effect.usage) {
        msg.inputTokens = effect.usage.inputTokens;
        msg.outputTokens = effect.usage.outputTokens;
        msg.cachedInputTokens = effect.usage.cachedInputTokens;
        msg.credits = effect.usage.credits;
        // 子智能体粒度用量按消息 id 归集（wire 消息无该字段，经 ctx map 供 VM 映射消费）
        if (effect.usage.subAgents?.length) {
          subAgentsByMessage.value[st.messageId] = effect.usage.subAgents;
        }
      }
      if (effect.status !== null) {
        msg.status = effect.status;
        if (effect.status === 2) autoSpeak(msg);
      }
      if (st.toolCalls.length) msg.toolCalls = st.toolCalls;
    }
    if (effect.pollAsync) void pollAsyncMessage(s.conversationId, st.messageId);
    streamingMessageId.value = null;
    finishSession(s);
  }

  /** 把纯归约产出的副作用解释执行为响应式写回 / rAF / 收尾（顺序即归约返回顺序） */
  function applyEffect(s: StreamSession, effect: StreamEffect) {
    const st = s.state;
    switch (effect.type) {
      case "bind": {
        let msg: AiMessageVO | null =
          s.message ?? findMessage(effect.from) ?? null;
        if (!msg) {
          // 兜底：占位可能已被并发消息加载覆盖，回退到流式中的 assistant 消息
          msg =
            messages.value.find(
              (item) => item.role === "assistant" && item.status === 1
            ) ?? null;
        }
        if (msg) {
          msg.id = st.messageId;
          s.message = msg;
        }
        return;
      }
      case "streaming-id":
        streamingMessageId.value = st.streamingMessageId;
        return;
      case "clear-interrupts":
        interrupts.value = [];
        interruptedMessageId.value = null;
        return;
      case "writing":
        if (st.thinking) thinkingByMessage.value[st.messageId] = st.thinking;
        return;
      case "thoughts":
        thoughtsByMessage.value[st.messageId] = st.thoughts;
        return;
      case "tool-calls":
        toolCallsByMessage.value[st.messageId] = st.toolCalls;
        return;
      case "plan":
        if (st.plan) planByMessage.value[st.messageId] = st.plan;
        return;
      case "interrupts":
        interrupts.value = st.interrupts;
        interruptedMessageId.value = st.interruptedMessageId;
        streamingMessageId.value = st.streamingMessageId;
        return;
      case "suggestions":
        suggestions.value = st.suggestions;
        return;
      case "flush":
        scheduleFlush(s);
        return;
      case "finish":
        finishSession(s);
        return;
      case "fail":
        failMessage(s, effect.reason);
        return;
      case "end":
        applyEnd(s, effect);
        return;
    }
  }

  /** 把归约结果落到会话：先替换纯状态，再按 effects 顺序驱动副作用 */
  function dispatch(s: StreamSession, event: StreamEvent, now: number) {
    const { state, effects } = reduceStreamState(s.state, event, now);
    s.state = state;
    for (const effect of effects) applyEffect(s, effect);
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

  /** SSE 回调 → 归约事件（network-error/close 属 transport 生命周期，直接交给重连处理） */
  function buildHandlers(s: StreamSession): MessageStreamHandlers {
    return {
      onStart(data) {
        const now = Date.now();
        s.lastTokenAt = now;
        if (data.streamSessionId) s.streamSessionId = data.streamSessionId;
        dispatch(s, { kind: "start", data }, now);
      },
      onContentBlockStart(data) {
        const now = Date.now();
        s.lastTokenAt = now;
        dispatch(s, { kind: "block-start", data }, now);
      },
      onContentBlockDelta(data) {
        const now = Date.now();
        s.lastTokenAt = now;
        dispatch(s, { kind: "delta", data }, now);
      },
      onContentBlockStop(data) {
        dispatch(s, { kind: "block-stop", data }, Date.now());
      },
      onThought(data) {
        const now = Date.now();
        s.lastTokenAt = now;
        dispatch(s, { kind: "thought", data }, now);
      },
      onPlan(data) {
        const now = Date.now();
        s.lastTokenAt = now;
        dispatch(s, { kind: "plan", data }, now);
      },
      onSuggestions(data) {
        dispatch(s, { kind: "suggestions", data }, Date.now());
      },
      onInterrupt(data) {
        dispatch(s, { kind: "interrupt", data }, Date.now());
      },
      onEventId(id) {
        s.lastEventId = id;
      },
      onPing() {
        // 心跳仅保活，不重置 token 超时计时
      },
      onError(data) {
        dispatch(s, { kind: "error", data }, Date.now());
      },
      onEnd(data) {
        dispatch(s, { kind: "end", data }, Date.now());
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
   * 同一会话同一时间仅允许一个流式输出进行中；返回 false 表示未开启（会话忙），调用方据此保留可重试状态。
   * 初始归约状态由既有 store 数据播种（思考/工具/计划/中断），续流时延续累积而非从零开始。
   */
  function openStream(
    conversationId: number,
    messageId: number,
    opener: (handlers: MessageStreamHandlers) => AbortController | void,
    options: { clearInterruptOnStart?: boolean } = {}
  ) {
    if (ctx.session) {
      ElMessage.warning("当前会话正在推理中，请等待完成或先停止");
      return false;
    }
    const s: StreamSession = {
      conversationId,
      controller: null,
      message: findMessage(messageId) ?? null,
      state: {
        messageId,
        textBuffer: "",
        thinking: null,
        thinkingBuffer: "",
        toolBlocks: new Map(),
        toolCalls: toolCallsByMessage.value[messageId] ?? [],
        thoughts: thoughtsByMessage.value[messageId] ?? [],
        plan: planByMessage.value[messageId],
        interrupts: interrupts.value.slice(),
        suggestions: suggestions.value.slice(),
        streamingMessageId: streamingMessageId.value,
        interruptedMessageId: interruptedMessageId.value,
        clearInterruptOnStart: options.clearInterruptOnStart,
      },
      flushScheduled: false,
      lastTokenAt: Date.now(),
      reconnectAttempts: 0,
      reconnectTimer: null,
      timeoutTimer: null,
      finished: false,
    };
    s.controller = opener(buildHandlers(s)) ?? null;
    ctx.session = s;
    startTimeoutWatch(s);
    return true;
  }

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
      id: nextLocalMessageId(),
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

  /**
   * 按中断子类型把调用方意图归一为 resume 载荷：
   * - plan_approve：透传计划干预 planEdit（wire 与 VM 同形，无需翻译）
   * - confirm(algorithm_recommend)：{confirm, params:{algorithmId}}（algorithmId 取调用方选择，缺省回退推荐值）
   * - confirm(tool_permission/dangerous_op/write_conflict)：{confirm}
   * - 其余（quota/async_wait）与无待处理中断：透传调用方载荷
   * 未知 confirmKind 由 resolveConfirmKind 抛错（不静默兜底）。
   */
  function buildResumeForm(
    pending: InterruptEvent | undefined,
    input: MessageResumeForm
  ): MessageResumeForm {
    if (!pending) return { ...input };
    if (pending.type === "plan_approve") {
      return input.planEdit !== undefined ? { planEdit: input.planEdit } : {};
    }
    if (pending.type !== "confirm") return { ...input };
    const kind = resolveConfirmKind(pending);
    const confirm = input.confirm ?? true;
    if (kind === "algorithm_recommend") {
      const selected = input.params?.algorithmId;
      const algorithmId =
        typeof selected === "number"
          ? selected
          : pending.data.recommendation?.algorithmId;
      return {
        confirm,
        ...(algorithmId !== undefined ? { params: { algorithmId } } : {}),
      };
    }
    return { confirm };
  }

  function resumeInterrupt(messageId: number, data: MessageResumeForm = {}) {
    const conversationId = currentConversationId.value;
    if (!conversationId) return;
    const form = buildResumeForm(interrupts.value[0], data);
    const msg = findMessage(messageId);
    if (msg) msg.status = 1;
    streamingMessageId.value = messageId;
    // 打开失败（会话忙）时保留中断点，用户可以再次调用 resume 重试
    const started = openStream(
      conversationId,
      messageId,
      (handlers) => AiConversationAPI.resumeMessage(messageId, form, handlers),
      { clearInterruptOnStart: true }
    );
    if (!started) streamingMessageId.value = null;
  }

  async function stopStreaming() {
    const messageId = streamingMessageId.value ?? interruptedMessageId.value;
    if (!messageId) return;
    ctx.session?.controller?.abort();
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
    if (ctx.session) finishSession(ctx.session);
    asyncPollToken++;
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
    ctx.session?.controller?.abort();
    finishSessionPartial();
    streamingMessageId.value = null;
    interruptedMessageId.value = null;
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
