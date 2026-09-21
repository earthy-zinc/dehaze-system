<!-- 用户消息：右侧气泡，支持复制/引用/编辑重发；管理端审计只读（纯展示，复制经 emit 交由宿主） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatScopeVM, ChatUserMessageVM } from "../types";

defineOptions({ name: "UserMessage" });

const props = defineProps<{
  message: ChatUserMessageVM;
  scope: ChatScopeVM;
}>();

const emit = defineEmits<{
  edit: [message: ChatUserMessageVM];
  copy: [message: ChatUserMessageVM];
  quote: [message: ChatUserMessageVM];
}>();

const readOnly = computed(() => props.scope === "admin");
</script>

<template>
  <div class="user-message">
    <div class="user-message__bubble">
      <div class="user-message__edited">
        <el-tag v-if="message.edited" size="small" type="info">
          已编辑
          <el-tooltip
            v-if="message.originalContent"
            :content="message.originalContent"
          >
            <span class="user-message__original">?</span>
          </el-tooltip>
        </el-tag>
      </div>
      <div class="user-message__content">{{ message.content }}</div>
    </div>
    <div v-if="!readOnly" class="user-message__actions">
      <el-button link size="small" @click="emit('copy', message)"
        >复制</el-button
      >
      <el-button link size="small" @click="emit('quote', message)"
        >引用</el-button
      >
      <el-button link size="small" @click="emit('edit', message)"
        >编辑</el-button
      >
    </div>
  </div>
</template>

<style scoped lang="scss">
.user-message {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-bottom: 16px;

  &__bubble {
    max-width: 72%;
    padding: 10px 14px;
    background-color: var(--el-color-primary-light-9);
    border-radius: 12px 12px 2px;
  }

  &__content {
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }

  &__edited {
    margin-bottom: 4px;
  }

  &__original {
    margin-left: 2px;
    cursor: help;
  }

  &__actions {
    margin-top: 2px;
    opacity: 0;
    transition: opacity 0.2s;
  }

  &:hover &__actions {
    opacity: 1;
  }
}
</style>
