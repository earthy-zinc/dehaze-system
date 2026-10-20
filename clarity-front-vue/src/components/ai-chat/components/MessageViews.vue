<!-- 消息数据视图层容器：纯透传（不读写任何 store），供用户端/管理端页面嵌入 -->
<script lang="ts" setup>
import MessageList from "./MessageList.vue";
import type {
  ChatArtifactVM,
  ChatAssistantMessageVM,
  ChatFeedbackVM,
  ChatInterruptVM,
  ChatMessageVM,
  ChatResumeFormVM,
  ChatScopeVM,
  ChatUserMessageVM,
} from "../types";

defineOptions({ name: "MessageViews" });

defineProps<{
  scope: ChatScopeVM;
  messages: ChatMessageVM[];
  streamingMessageId: number | null;
  interruptedMessageId: number | null;
  interrupts: ChatInterruptVM[];
  scrollFollowEnabled: boolean;
  hasMoreHistory?: boolean;
  loadingMore?: boolean;
}>();

const emit = defineEmits<{
  "scroll-follow-toggle": [enabled: boolean];
  "reach-top": [];
  edit: [message: ChatUserMessageVM];
  copy: [message: ChatMessageVM];
  quote: [message: ChatMessageVM];
  regenerate: [message: ChatAssistantMessageVM];
  feedback: [message: ChatAssistantMessageVM, data: ChatFeedbackVM | null];
  delete: [message: ChatMessageVM];
  speak: [message: ChatAssistantMessageVM];
  trace: [message: ChatMessageVM];
  "resume-interrupt": [messageId: number, data: ChatResumeFormVM];
  "open-artifact": [artifact: ChatArtifactVM];
  "apply-suggestion": [question: string];
  "load-artifacts": [messageId: number];
  retry: [message: ChatAssistantMessageVM];
}>();
</script>

<template>
  <MessageList
    :messages="messages"
    :scope="scope"
    :streaming-message-id="streamingMessageId"
    :interrupted-message-id="interruptedMessageId"
    :interrupts="interrupts"
    :scroll-follow-enabled="scrollFollowEnabled"
    :has-more-history="hasMoreHistory"
    :loading-more="loadingMore"
    @scroll-follow-toggle="(enabled) => emit('scroll-follow-toggle', enabled)"
    @reach-top="emit('reach-top')"
    @edit="(m) => emit('edit', m)"
    @copy="(m) => emit('copy', m)"
    @quote="(m) => emit('quote', m)"
    @regenerate="(m) => emit('regenerate', m)"
    @feedback="(m, data) => emit('feedback', m, data)"
    @delete="(m) => emit('delete', m)"
    @speak="(m) => emit('speak', m)"
    @trace="(m) => emit('trace', m)"
    @resume-interrupt="(id, data) => emit('resume-interrupt', id, data)"
    @open-artifact="(a) => emit('open-artifact', a)"
    @apply-suggestion="(q) => emit('apply-suggestion', q)"
    @load-artifacts="(id) => emit('load-artifacts', id)"
    @retry="(m) => emit('retry', m)"
  />
</template>
