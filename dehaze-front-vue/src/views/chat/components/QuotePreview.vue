<!-- 引用预览条：展示被引用消息摘要，可移除；发送时由 chatStore 合并进消息内容 -->
<script lang="ts" setup>
import { ChatDotRound, Close } from "@element-plus/icons-vue";
import { computed } from "vue";
import { useChatStore } from "@/store/modules/chat";

defineOptions({ name: "QuotePreview" });

const chatStore = useChatStore();

const summary = computed(() => {
  const content = chatStore.quotedMessage?.content ?? "";
  return content.length > 100 ? `${content.slice(0, 100)}…` : content;
});

function removeQuote() {
  chatStore.quotedMessage = null;
}
</script>

<template>
  <div v-if="chatStore.quotedMessage" class="quote-preview">
    <el-icon><ChatDotRound /></el-icon>
    <span class="quote-preview__text">{{ summary }}</span>
    <el-button link size="small" :icon="Close" @click="removeQuote" />
  </div>
</template>

<style scoped lang="scss">
.quote-preview {
  display: flex;
  gap: 6px;
  align-items: center;
  padding: 6px 10px;
  margin: 0 16px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  background-color: var(--el-fill-color-light);
  border-radius: 6px;

  &__text {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}
</style>
