<!-- 消息数据视图层容器：按 scope 渲染消息流，供用户端/管理端页面嵌入 -->
<script lang="ts" setup>
import type { AiMessageVO, FeedbackForm } from "dehaze-sdk-js";
import { storeToRefs } from "pinia";
import { useChatStore, type ChatScope } from "@/store/modules/chat";
import MessageList from "./MessageList.vue";

defineOptions({ name: "MessageViews" });

defineProps<{
  scope: ChatScope;
  conversationId?: number;
  messages: AiMessageVO[];
}>();

const emit = defineEmits<{
  "reach-bottom": [];
  edit: [message: AiMessageVO];
  copy: [];
  quote: [message: AiMessageVO];
  regenerate: [message: AiMessageVO];
  feedback: [message: AiMessageVO, data: FeedbackForm | null];
  delete: [message: AiMessageVO];
  speak: [message: AiMessageVO];
  trace: [message: AiMessageVO];
}>();

const chatStore = useChatStore();
const { streamingMessageId, scrollFollowEnabled } = storeToRefs(chatStore);
</script>

<template>
  <MessageList
    :messages="messages"
    :streaming-message-id="streamingMessageId"
    :scroll-follow-enabled="scrollFollowEnabled"
    @scroll-follow-toggle="chatStore.scrollFollowEnabled = $event"
    @reach-bottom="emit('reach-bottom')"
    @edit="(m) => emit('edit', m)"
    @copy="emit('copy')"
    @quote="(m) => emit('quote', m)"
    @regenerate="(m) => emit('regenerate', m)"
    @feedback="(m, data) => emit('feedback', m, data)"
    @delete="(m) => emit('delete', m)"
    @speak="(m) => emit('speak', m)"
    @trace="(m) => emit('trace', m)"
  />
</template>
