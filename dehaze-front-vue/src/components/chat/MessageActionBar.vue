<!-- 消息操作栏（悬停出现）：复制/引用/重新生成/朗读/删除/反馈；管理端审计不提供内容操作 -->
<script lang="ts" setup>
import type { AiMessageVO } from "dehaze-sdk-js";
import { computed } from "vue";
import type { ChatScope } from "@/store/modules/chat";

defineOptions({ name: "MessageActionBar" });

const props = defineProps<{
  message: AiMessageVO;
  scope: ChatScope;
}>();

const emit = defineEmits<{
  copy: [];
  quote: [message: AiMessageVO];
  regenerate: [message: AiMessageVO];
  speak: [message: AiMessageVO];
  delete: [message: AiMessageVO];
  feedback: [message: AiMessageVO];
}>();

const isAssistant = computed(() => props.message.role === "assistant");
const interactive = computed(
  () => props.scope === "self" && props.message.status >= 2
);
</script>

<template>
  <div v-if="scope === 'self'" class="message-action-bar">
    <el-button link size="small" @click="emit('copy')">复制</el-button>
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
