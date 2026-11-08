<!-- 用户消息：右侧气泡，支持复制/引用/编辑重发；管理端审计只读（纯展示，复制经 emit 交由宿主） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { Message } from "clarity-sdk-js";
import { textOfBlocks } from "../model/blocks";
import type { ChatMessageActionVM, ChatScopeVM } from "../types";

defineOptions({ name: "UserMessage" });

const props = defineProps<{
  message: Message;
  scope: ChatScopeVM;
}>();

const emit = defineEmits<{
  action: [action: ChatMessageActionVM];
}>();

function emitAction(type: "copy" | "quote" | "edit") {
  emit("action", { type, messageId: props.message.id });
}

const readOnly = computed(() => props.scope === "admin");
const text = computed(() => textOfBlocks(props.message.blocks));
</script>

<template>
  <div class="user-message">
    <div class="user-message__bubble">
      <div class="user-message__edited">
        <el-tag v-if="message.edited" size="small" type="info">已编辑</el-tag>
      </div>
      <div class="user-message__content">{{ text }}</div>
    </div>
    <div v-if="!readOnly" class="user-message__actions">
      <el-button link size="small" @click="emitAction('copy')">复制</el-button>
      <el-button link size="small" @click="emitAction('quote')">引用</el-button>
      <el-button link size="small" @click="emitAction('edit')">编辑</el-button>
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
