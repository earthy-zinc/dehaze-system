<!-- 消息操作栏（悬停出现）：复制/引用/重新生成/朗读/删除/反馈；管理端审计不提供内容操作（纯展示） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatMessageStatusVM, ChatMessageVM, ChatScopeVM } from "../types";

defineOptions({ name: "MessageActionBar" });

const props = defineProps<{
  message: ChatMessageVM;
  scope: ChatScopeVM;
}>();

const emit = defineEmits<{
  copy: [message: ChatMessageVM];
  quote: [message: ChatMessageVM];
  regenerate: [message: ChatMessageVM];
  speak: [message: ChatMessageVM];
  delete: [message: ChatMessageVM];
  feedback: [message: ChatMessageVM];
}>();

const isAssistant = computed(() => props.message.role === "assistant");

const assistantStatus = computed<ChatMessageStatusVM | undefined>(() =>
  props.message.role === "assistant" ? props.message.status : undefined
);

// 仅用户端且助手消息已到终态才提供内容操作入口
const interactive = computed(
  () =>
    props.scope === "self" &&
    isAssistant.value &&
    assistantStatus.value !== "streaming"
);
</script>

<template>
  <div v-if="scope === 'self'" class="message-action-bar">
    <el-button link size="small" @click="emit('copy', message)">复制</el-button>
    <template v-if="isAssistant && interactive">
      <el-button link size="small" @click="emit('quote', message)"
        >引用</el-button
      >
      <el-button link size="small" @click="emit('regenerate', message)"
        >重新生成</el-button
      >
      <el-button link size="small" @click="emit('speak', message)"
        >朗读</el-button
      >
      <el-button
        link
        size="small"
        type="danger"
        @click="emit('delete', message)"
        >删除</el-button
      >
      <el-button link size="small" @click="emit('feedback', message)"
        >反馈</el-button
      >
    </template>
  </div>
</template>

<style scoped lang="scss">
.message-action-bar {
  display: flex;
  gap: 4px;
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;

  .message-block:hover & {
    opacity: 1;
  }
}
</style>
