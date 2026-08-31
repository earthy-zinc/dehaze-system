<!-- 对话头部：会话标题 + 模型切换（下一条消息生效）+ 导出 + 设置/记忆/定时任务入口 -->
<script lang="ts" setup>
import { computed } from "vue";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "ChatHeader" });

const emit = defineEmits<{
  "open-settings": [];
  "open-memory": [];
  "open-task": [];
}>();

const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

const currentConversation = computed(() =>
  chatStore.conversations.find(
    (item) => item.id === chatStore.currentConversationId
  )
);

const currentModel = computed({
  get: () => currentConversation.value?.model ?? "",
  set: (model: string) => {
    if (currentConversation.value && model) {
      chatStore.updateConversation(currentConversation.value.id, { model });
    }
  },
});

function handleExport(format: "markdown" | "json") {
  if (!currentConversation.value) return;
  chatStore.exportConversation(currentConversation.value.id, format);
}
</script>

<template>
  <div class="chat-header">
    <div class="chat-header__title">
      {{ currentConversation?.title ?? "新对话" }}
    </div>

    <el-select
      v-model="currentModel"
      placeholder="选择模型"
      class="chat-header__model"
      size="small"
    >
      <el-option
        v-for="model in chatUserStore.availableModels"
        :key="model.modelId"
        :label="model.displayName"
        :value="model.modelId"
      >
        <span>{{ model.displayName }}</span>
        <el-tag v-if="model.supportsMultimodal === 1" size="small" class="ml-2">
          多模态
        </el-tag>
      </el-option>
    </el-select>

    <div class="chat-header__actions">
      <el-dropdown trigger="click" @command="handleExport">
        <el-button size="small" :disabled="!currentConversation"
          >导出</el-button
        >
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="markdown">Markdown</el-dropdown-item>
            <el-dropdown-item command="json">JSON</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
      <el-button
        size="small"
        :disabled="!currentConversation"
        @click="emit('open-task')"
      >
        保存为定时任务
      </el-button>
      <el-button size="small" @click="emit('open-memory')">长期记忆</el-button>
      <el-button
        size="small"
        type="primary"
        plain
        :disabled="!currentConversation"
        @click="emit('open-settings')"
      >
        会话设置
      </el-button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.chat-header {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 10px 16px;
  background-color: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-light);

  &__title {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 15px;
    font-weight: 600;
    white-space: nowrap;
  }

  &__model {
    width: 220px;
  }

  &__actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }
}
</style>
