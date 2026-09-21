// AI 对话公共 Store 门面：用户端（scope=self）与管理端会话审计（scope=admin）共用的消息数据视图层
// 拆分为 conversation/messages/stream/feedback 四个 slice，共享同一份 ctx 状态；对外 API 与拆分前逐项一致
import {
  type AiMessageVO,
  type ConversationVO,
  type FeedbackVO,
  type InterruptEvent,
  type MemoryVO,
  type ThoughtEvent,
} from "dehaze-sdk-js";
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { createConversationSlice } from "./conversation";
import { createFeedbackSlice } from "./feedback";
import { createMessagesSlice } from "./messages";
import type { ChatPlanVM } from "@/components/ai-chat/types";
import type {
  ChatCtx,
  ChatScope,
  SubAgentUsage,
  ThinkingState,
} from "./shared";
import { resolveConfirmKind } from "./shared";
import { createStreamSlice } from "./stream";

export { resolveConfirmKind };
export type {
  BatchAction,
  ChatScope,
  ConversationFilterStatus,
  StreamSession,
  SubAgentUsage,
  ThinkingSegment,
  ThinkingState,
  ToolCallDraft,
} from "./shared";

export const useChatStore = defineStore("chat", () => {
  const scope = ref<ChatScope>("self");

  // ===== 会话列表 =====
  const conversations = ref<ConversationVO[]>([]);

  // ===== 当前会话与消息 =====
  const currentConversationId = ref<number | null>(null);
  const messages = ref<AiMessageVO[]>([]);

  // ===== 流式状态 =====
  const streamingMessageId = ref<number | null>(null);
  const interruptedMessageId = ref<number | null>(null);
  const interrupts = ref<InterruptEvent[]>([]);
  const suggestions = ref<string[]>([]);
  const thoughtsByMessage = ref<Record<number, ThoughtEvent[]>>({});
  const thinkingByMessage = ref<Record<number, ThinkingState>>({});
  const toolCallsByMessage = ref<Record<number, unknown[]>>({});
  const planByMessage = ref<Record<number, ChatPlanVM>>({});
  const messageMemories = ref<Record<number, MemoryVO[]>>({});
  const subAgentsByMessage = ref<Record<number, SubAgentUsage>>({});

  // ===== 本地交互状态 =====
  const selectionMode = ref(false);
  const scrollFollowEnabled = ref(true);
  const quotedMessage = ref<AiMessageVO | null>(null);
  const feedbackByMessage = ref<Record<number, FeedbackVO | null>>({});

  const isStreaming = computed(
    () =>
      streamingMessageId.value !== null || interruptedMessageId.value !== null
  );

  // 共享上下文：先创建全部状态，再注入各 slice 工厂，运行时经 ctx 互相引用以打破环依赖
  const ctx: ChatCtx = {
    scope,
    conversations,
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
    messageMemories,
    subAgentsByMessage,
    selectionMode,
    scrollFollowEnabled,
    quotedMessage,
    feedbackByMessage,
    isStreaming,
    session: null,
  };

  // stream slice 先建（messages/conversation 需要其 teardownStream 注入）
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
    branchSwitching,
    fetchMessages,
    loadMoreMessages,
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

  const { submitFeedback, fetchFeedback } = feedback;

  return {
    scope,
    conversations,
    conversationsTotal,
    conversationsLoading,
    conversationQuery,
    currentConversationId,
    messages,
    messagesLoading,
    branchSwitching,
    streamingMessageId,
    interruptedMessageId,
    interrupts,
    suggestions,
    thoughtsByMessage,
    thinkingByMessage,
    toolCallsByMessage,
    planByMessage,
    messageMemories,
    subAgentsByMessage,
    selectionMode,
    scrollFollowEnabled,
    quotedMessage,
    feedbackByMessage,
    isStreaming,
    messagesLoadingMore,
    messagesHasMore,
    initScope,
    fetchConversations,
    createConversation,
    updateConversation,
    deleteConversation,
    batchOperateConversations,
    exportConversation,
    fetchMessages,
    loadMoreMessages,
    markConversationRead,
    switchBranch,
    sendMessage,
    regenerate,
    editMessage,
    resumeInterrupt,
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
