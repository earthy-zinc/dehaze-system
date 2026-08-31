<!-- 消息输入区：回车发送/Shift+换行、按会话草稿持久化、语音输入、图片附件（校验多模态能力）、流式中变停止按钮 -->
<script lang="ts" setup>
import { Picture, Promotion, VideoPause } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import { FileAPI } from "dehaze-sdk-js";
import { watchDebounced } from "@vueuse/core";
import { computed, ref, watch } from "vue";
import { useChatStore } from "@/store/modules/chat";
import { useChatUserStore } from "@/store/modules/chatUser";

defineOptions({ name: "MessageInput" });

const emit = defineEmits<{
  send: [text: string];
  stop: [];
}>();

const chatStore = useChatStore();
const chatUserStore = useChatUserStore();

const text = ref("");
const uploading = ref(false);
const fileInputRef = ref<HTMLInputElement>();

// 切换会话时恢复对应草稿
watch(
  () => chatStore.currentConversationId,
  (id) => {
    text.value = chatUserStore.getDraft(id);
  },
  { immediate: true }
);

// 草稿防抖持久化，刷新/误关闭不丢失
watchDebounced(
  text,
  (value) => chatUserStore.saveDraft(chatStore.currentConversationId, value),
  { debounce: 300 }
);

const currentModel = computed(() => {
  const model = chatStore.conversations.find(
    (item) => item.id === chatStore.currentConversationId
  )?.model;
  return chatUserStore.availableModels.find((item) => item.modelId === model);
});

function handleSend() {
  const content = text.value.trim();
  if (!content) return;
  if (chatStore.isStreaming) {
    ElMessage.warning("当前正在推理中，请先停止或等待完成");
    return;
  }
  emit("send", content);
  text.value = "";
}

// 附件上传前校验当前模型多模态能力；会话未指定模型时交由后端判定
function handleAttachClick() {
  if (currentModel.value && currentModel.value.supportsMultimodal !== 1) {
    ElMessage.warning("当前模型不支持图片输入，请先切换多模态模型");
    return;
  }
  fileInputRef.value?.click();
}

async function handleFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    ElMessage.warning("仅支持图片附件");
    return;
  }
  uploading.value = true;
  try {
    const info = await FileAPI.upload(file);
    text.value = `${text.value}${text.value ? "\n" : ""}![${info.name}](${info.url})`;
  } finally {
    uploading.value = false;
  }
}
</script>

<template>
  <div class="message-input">
    <QuotePreview />

    <div class="message-input__toolbar">
      <VoiceInput v-model="text" :disabled="chatStore.isStreaming" />
      <el-tooltip content="图片附件" placement="top">
        <el-button
          circle
          :icon="Picture"
          :loading="uploading"
          :disabled="chatStore.isStreaming"
          @click="handleAttachClick"
        />
      </el-tooltip>
      <input
        ref="fileInputRef"
        type="file"
        accept="image/*"
        class="message-input__file"
        @change="handleFileChange"
      />
    </div>

    <div class="message-input__row">
      <el-input
        v-model="text"
        type="textarea"
        :autosize="{ minRows: 2, maxRows: 6 }"
        resize="none"
        placeholder="输入消息，Enter 发送，Shift + Enter 换行"
        @keydown.enter.exact.prevent="handleSend"
      />
      <el-button
        v-if="chatStore.isStreaming"
        type="danger"
        :icon="VideoPause"
        @click="emit('stop')"
      >
        停止
      </el-button>
      <el-button
        v-else
        type="primary"
        :icon="Promotion"
        :disabled="!text.trim() || uploading"
        @click="handleSend"
      >
        发送
      </el-button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.message-input {
  padding: 8px 16px 12px;
  background-color: var(--el-bg-color);
  border-top: 1px solid var(--el-border-color-light);

  &__toolbar {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
  }

  &__file {
    display: none;
  }

  &__row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
}
</style>
