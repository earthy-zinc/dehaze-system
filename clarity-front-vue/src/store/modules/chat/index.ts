// AI 对话公共 Store 门面：用户端（scope=self）与管理端会话审计（scope=admin）共用的消息数据视图层。
//
// 唯一内容源：messages: Record<id, Message>（契约 Message，blocks 承载全部内容）。
// 全局流式态只有 streamStatus / streamMessageId / streamSessionId 三项；
// 计划、中断、用量、推荐问题、产物/记忆引用全部挂在消息实体上，没有侧表。
import {
  type ConversationVO,
  type Feedback,
  type Message,
} from "clarity-sdk-js";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { assertNever } from "@/utils/assertNever";
import { createConversationSlice } from "./conversation";
import { createFeedbackSlice } from "./feedback";
import { createMessagesSlice } from "./messages";
import type { ChatCtx, ChatScope, MessageAction, StreamStatus } from "./shared";
import { resolveConfirmKind } from "./shared";
import { createStreamSlice } from "./stream";

export { resolveConfirmKind };
export type { MessagesError } from "./messages";
export type {
  BatchAction,
  ChatScope,
  ConversationFilterStatus,
  MessageAction,
  StreamSession,
  StreamStatus,
} from "./shared";

export const useChatStore = defineStore("chat", () => {
  const scope = ref<ChatScope>("self");

  // ===== 会话列表 =====
  const conversations = ref<ConversationVO[]>([]);

  // ===== 当前会话与消息实体 =====
  const currentConversationId = ref<number | null>(null);
  /** 消息实体表（id → 契约 Message），唯一内容源 */
  const messages = ref<Record<number, Message>>({});
  /** 消息顺序（id 升序；本地负数占位在末尾） */
  const messageOrder = ref<number[]>([]);

  // ===== 全局流式态（仅此三项） =====
  const streamStatus = ref<StreamStatus>("idle");
  const streamMessageId = ref<number | null>(null);
  const streamSessionId = ref<string | null>(null);

  // ===== 本地交互状态 =====
  const selectionMode = ref(false);
  const scrollFollowEnabled = ref(true);
  const quotedMessage = ref<Message | null>(null);
  const feedbackByMessage = ref<Record<number, Feedback | null>>({});

  /** 消息顺序 → 实体数组（渲染与投影的输入） */
  const messageList = computed<Message[]>(() =>
    messageOrder.value
      .map((id) => messages.value[id])
      .filter((message): message is Message => Boolean(message))
  );

  // 共享上下文：先创建全部状态，再注入各 slice 工厂，运行时经 ctx 互相引用以打破环依赖
  const ctx: ChatCtx = {
    scope,
    conversations,
    messages,
    messageOrder,
    currentConversationId,
    streamStatus,
    streamMessageId,
    streamSessionId,
    selectionMode,
    scrollFollowEnabled,
    quotedMessage,
    feedbackByMessage,
    session: null,
  };

  // stream slice 先建（conversation/messages 需要其 teardownStream 注入）
  const stream = createStreamSlice(ctx);
  const conversation = createConversationSlice(ctx, stream.teardownStream);
  const messagesSlice = createMessagesSlice(ctx, stream.teardownStream);
  const feedback = createFeedbackSlice(ctx);

  const {
    conversationsTotal,
    conversationsLoading,
    conversationQuery,
    initScope,
    fetchConversations,
    createConversation,
    updateConversation,
    deleteConversation,
    batchOperateConversations,
    exportConversation,
  } = conversation;

  const {
    messagesLoading,
    messagesLoadingMore,
    messagesHasMore,
    messagesError,
    branchSwitching,
    fetchMessages,
    loadMoreMessages,
    retryMessages,
    deleteMessage,
    markConversationRead,
    switchBranch,
  } = messagesSlice;

  const {
    sendMessage,
    regenerate,
    editMessage,
    resumeInterrupt,
    stopStreaming,
    quoteMessage,
    speakMessage,
    applySuggestion,
    teardownStream,
  } = stream;

  const { submitFeedback, applyConversationFeedback } = feedback;

  /**
   * 单一消息动作入口（判别联合）：组件层只上抛一种 action 事件，绑定层原样转交。
   * 复制/编辑由页面层就地处理，不落到 store。
   */
  async function dispatchMessageAction(action: MessageAction): Promise<void> {
    switch (action.type) {
      case "regenerate":
      case "retry":
        regenerate(action.messageId);
        return;
      case "delete":
        await deleteMessage(action.messageId);
        return;
      case "quote":
        quoteMessage(action.messageId);
        return;
      case "speak":
        await speakMessage(action.messageId);
        return;
      case "feedback":
        await submitFeedback(action.messageId, action.data);
        return;
      case "resume-interrupt":
        resumeInterrupt(action.messageId, action.data);
        return;
      case "apply-suggestion":
        applySuggestion(action.question);
        return;
      default:
        // 类型上不可达：MessageAction 新增成员时实参不再是 never，此处编译失败。
        // （copy/edit 已被 MessageAction 排除，由页面层就地处理）
        return assertNever(action);
    }
  }

  return {
    scope,
    conversations,
    conversationsTotal,
    conversationsLoading,
    conversationQuery,
    currentConversationId,
    messages,
    messageOrder,
    messageList,
    streamStatus,
    streamMessageId,
    streamSessionId,
    selectionMode,
    scrollFollowEnabled,
    quotedMessage,
    feedbackByMessage,
    messagesLoading,
    messagesLoadingMore,
    messagesHasMore,
    messagesError,
    branchSwitching,
    initScope,
    fetchConversations,
    createConversation,
    updateConversation,
    deleteConversation,
    batchOperateConversations,
    exportConversation,
    fetchMessages,
    loadMoreMessages,
    retryMessages,
    markConversationRead,
    switchBranch,
    sendMessage,
    regenerate,
    editMessage,
    resumeInterrupt,
    stopStreaming,
    submitFeedback,
    applyConversationFeedback,
    deleteMessage,
    quoteMessage,
    speakMessage,
    applySuggestion,
    teardownStream,
    dispatchMessageAction,
  };
});
