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

/**
 * 顶栏「···」菜单分发：顶栏只留高频的会话标题与模型选择，
 * 导出/定时任务/长期记忆/会话设置均为低频操作，统一收进菜单（设计评审 #4）
 */
function handleAction(command: string) {
  switch (command) {
    case "settings":
      emit("open-settings");
      break;
    case "memory":
      emit("open-memory");
      break;
    case "task":
      emit("open-task");
      break;
    case "export-markdown":
      handleExport("markdown");
      break;
    case "export-json":
      handleExport("json");
      break;
  }
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
      <el-dropdown trigger="click" @command="handleAction">
        <el-button size="small" class="chat-header__more">···</el-button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="memory">长期记忆</el-dropdown-item>
            <el-dropdown-item
              command="settings"
              :disabled="!currentConversation"
            >
              会话设置
            </el-dropdown-item>
            <el-dropdown-item command="task" :disabled="!currentConversation">
              保存为定时任务
            </el-dropdown-item>
            <el-dropdown-item
              command="export-markdown"
              :disabled="!currentConversation"
              divided
            >
              导出 Markdown
            </el-dropdown-item>
            <el-dropdown-item
              command="export-json"
              :disabled="!currentConversation"
            >
              导出 JSON
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
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

  &__more {
    letter-spacing: 1px;
  }
}
</style>
