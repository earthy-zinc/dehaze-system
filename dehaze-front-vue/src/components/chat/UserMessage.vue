<!-- 用户消息：右侧气泡，支持复制/引用/编辑重发；管理端审计只读 -->
<script lang="ts" setup>
import { InfoFilled } from "@element-plus/icons-vue";
import type { AiMessageVO } from "dehaze-sdk-js";
import { computed } from "vue";
import { ElMessage } from "element-plus";
import { useChatStore } from "@/store/modules/chat";

defineOptions({ name: "UserMessage" });

const props = defineProps<{
  message: AiMessageVO;
}>();

const emit = defineEmits<{
  edit: [message: AiMessageVO];
  copy: [];
  quote: [message: AiMessageVO];
}>();

const chatStore = useChatStore();
const readOnly = computed(() => chatStore.scope === "admin");

async function handleCopy() {
  await navigator.clipboard.writeText(props.message.content ?? "");
  ElMessage.success("已复制");
  emit("copy");
}
</script>

<template>
  <div class="user-message">
    <div class="user-message__bubble">
      <div class="user-message__edited">
        <el-tag v-if="message.edited === 1" size="small" type="info">
          已编辑
          <el-tooltip
            v-if="message.originalContent"
            :content="message.originalContent"
          >
            <el-icon><InfoFilled /></el-icon>
          </el-tooltip>
        </el-tag>
      </div>
      <div class="user-message__content">{{ message.content }}</div>
    </div>
    <div v-if="!readOnly" class="user-message__actions">
      <el-button link size="small" @click="handleCopy">复制</el-button>
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
