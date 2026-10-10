<!-- 消息操作栏（悬停出现）：默认只露「复制」，其余操作收进「···」菜单
     （7 个按钮平铺会让操作栏比正文更抢眼）；管理端审计不提供内容操作（纯展示） -->
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

/** 「···」菜单项 → 事件映射（按事件分别 emit，保留 emit 的类型检查） */
function handleCommand(command: string): void {
  const handlers: Record<string, () => void> = {
    quote: () => emit("quote", props.message),
    regenerate: () => emit("regenerate", props.message),
    speak: () => emit("speak", props.message),
    feedback: () => emit("feedback", props.message),
    delete: () => emit("delete", props.message),
  };
  handlers[command]?.();
}
</script>

<template>
  <div v-if="scope === 'self'" class="message-action-bar">
    <el-button link size="small" @click="emit('copy', message)">复制</el-button>
    <el-dropdown v-if="interactive" trigger="click" @command="handleCommand">
      <el-button link size="small" class="more-actions">···</el-button>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item command="quote">引用</el-dropdown-item>
          <el-dropdown-item command="regenerate">重新生成</el-dropdown-item>
          <el-dropdown-item command="speak">朗读</el-dropdown-item>
          <el-dropdown-item command="feedback">反馈</el-dropdown-item>
          <el-dropdown-item command="delete" divided>删除</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
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

  .more-actions {
    letter-spacing: 1px;
  }
}
</style>
