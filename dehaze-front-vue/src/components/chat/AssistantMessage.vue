<!-- 助手消息：聚合 Markdown/思考过程/推理链/产物/记忆引用/反馈/推荐问题/操作栏 -->
<script lang="ts" setup>
import {
  AiConversationAPI,
  type ArtifactVO,
  type AiMessageVO,
  type FeedbackForm,
  type FeedbackVO,
} from "dehaze-sdk-js";
import { computed, onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { storeToRefs } from "pinia";
import { useChatStore, type ChatScope } from "@/store/modules/chat";
import MarkdownRenderer from "../MarkdownRenderer.vue";
import ArtifactCard from "./ArtifactCard.vue";
import FeedbackBar from "./FeedbackBar.vue";
import MemoryReferenceList from "./MemoryReferenceList.vue";
import MessageActionBar from "./MessageActionBar.vue";
import SuggestionList from "./SuggestionList.vue";
import ThoughtChain from "./ThoughtChain.vue";

defineOptions({ name: "AssistantMessage" });

const props = defineProps<{
  message: AiMessageVO;
  scope: ChatScope;
}>();

const emit = defineEmits<{
  regenerate: [message: AiMessageVO];
  quote: [message: AiMessageVO];
  feedback: [message: AiMessageVO, data: FeedbackForm | null];
  delete: [message: AiMessageVO];
  speak: [message: AiMessageVO];
  copy: [];
  trace: [message: AiMessageVO];
}>();

const chatStore = useChatStore();
const { thoughtsByMessage, thinkingByMessage, feedbackByMessage } =
  storeToRefs(chatStore);

const thoughts = computed(
  () => thoughtsByMessage.value[props.message.id] ?? []
);
const thinkingList = computed(
  () => thinkingByMessage.value[props.message.id] ?? []
);
const feedback = computed<FeedbackVO | null>(
  () => feedbackByMessage.value[props.message.id] ?? null
);
const memories = computed(
  () => chatStore.messageMemories[props.message.id] ?? []
);

const isStreaming = computed(() => props.message.status === 1);
const isFailed = computed(() => props.message.status === 3);
const isCanceled = computed(() => props.message.status === 4);

// 推荐问题仅挂在最后一条 assistant 消息之后
const showSuggestions = computed(() => {
  if (chatStore.suggestions.length === 0) return false;
  const list = chatStore.messages;
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i].role === "assistant") return list[i].id === props.message.id;
  }
  return false;
});

const artifacts = ref<ArtifactVO[]>([]);
const artifactDetail = ref<Record<string, unknown> | null>(null);
const artifactDialogVisible = ref(false);
const feedbackEditing = ref(false);

onMounted(async () => {
  if (props.message.id > 0) {
    chatStore.fetchFeedback(props.message.id);
    if (props.message.status >= 2) {
      try {
        artifacts.value = await AiConversationAPI.getMessageArtifacts(
          props.message.id
        );
      } catch {
        // 产物加载失败不阻塞消息展示
      }
    }
  }
});

async function handleCopy() {
  await navigator.clipboard.writeText(props.message.content ?? "");
  ElMessage.success("已复制");
  emit("copy");
}

function handleFeedbackSubmit(data: FeedbackForm | null) {
  feedbackEditing.value = false;
  emit("feedback", props.message, data);
}

function handleFeedbackCancel() {
  feedbackEditing.value = false;
  // 已有反馈时 cancel 语义为撤销反馈
  if (feedback.value) {
    emit("feedback", props.message, null);
  }
}

async function openArtifact(artifact: ArtifactVO) {
  try {
    artifactDetail.value = await AiConversationAPI.getArtifactDetail(
      artifact.id
    );
    artifactDialogVisible.value = true;
  } catch {
    ElMessage.error("产物详情加载失败");
  }
}
</script>

<template>
  <div class="message-block assistant-message">
    <!-- 推理模型：分析过程与正式回复分段展示，默认折叠；多段思考各成一块 -->
    <el-collapse
      v-if="thinkingList.length"
      class="assistant-message__thinking"
    >
      <el-collapse-item
        v-for="(text, index) in thinkingList"
        :key="index"
        :name="`thinking-${index}`"
        :title="
          thinkingList.length > 1 ? `思考过程 ${index + 1}` : '思考过程'
        "
      >
        <div class="assistant-message__thinking-body">
          <MarkdownRenderer :content="text" />
        </div>
      </el-collapse-item>
    </el-collapse>

    <ThoughtChain v-if="thoughts.length > 0" :thoughts="thoughts" />

    <div class="assistant-message__bubble">
      <template v-if="isStreaming && !message.content">
        <span class="assistant-message__typing">正在思考…</span>
      </template>
      <MarkdownRenderer
        v-else-if="message.content"
        :content="message.content"
      />
      <el-alert
        v-if="isFailed"
        type="error"
        :closable="false"
        :title="message.error || '回复生成失败'"
      />
      <el-tag v-if="isCanceled" type="info" size="small">已停止生成</el-tag>
    </div>

    <ArtifactCard
      v-for="artifact in artifacts"
      :key="artifact.id"
      :artifact="artifact"
      @open="openArtifact"
    />

    <MemoryReferenceList v-if="memories.length > 0" :memories="memories" />

    <FeedbackBar
      v-if="feedbackEditing"
      :message-id="message.id"
      :feedback="feedback"
      @submit="handleFeedbackSubmit"
      @cancel="handleFeedbackCancel"
    />
    <div v-else-if="feedback" class="assistant-message__feedback-summary">
      <el-tag :type="feedback.rating === 1 ? 'success' : 'danger'" size="small">
        {{ feedback.rating === 1 ? "已点赞" : "已点踩" }}
      </el-tag>
      <el-button link size="small" @click="feedbackEditing = true"
        >修改</el-button
      >
    </div>

    <SuggestionList
      v-if="showSuggestions && !isStreaming"
      :questions="chatStore.suggestions"
      @apply="(question) => chatStore.applySuggestion(question)"
    />

    <div class="assistant-message__footer">
      <MessageActionBar
        :message="message"
        :scope="scope"
        @copy="handleCopy"
        @quote="(m) => emit('quote', m)"
        @regenerate="(m) => emit('regenerate', m)"
        @speak="(m) => emit('speak', m)"
        @delete="(m) => emit('delete', m)"
        @feedback="() => (feedbackEditing = true)"
      />
      <el-button
        v-if="scope === 'admin'"
        link
        size="small"
        type="primary"
        @click="emit('trace', message)"
      >
        链路下钻
      </el-button>
    </div>

    <el-dialog v-model="artifactDialogVisible" title="产物详情" width="640px">
      <pre v-if="artifactDetail" class="assistant-message__artifact-detail">{{
        JSON.stringify(artifactDetail, null, 2)
      }}</pre>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.assistant-message {
  margin-bottom: 16px;

  &__thinking {
    margin-bottom: 8px;

    :deep(.el-collapse-item__header) {
      height: 32px;
      font-size: 13px;
      color: var(--el-text-color-secondary);
    }
  }

  &__thinking-body {
    padding: 8px;
    font-size: 13px;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }

  &__bubble {
    max-width: 92%;
    padding: 10px 14px;
    background-color: var(--el-fill-color-light);
    border-radius: 2px 12px 12px;
  }

  &__typing {
    color: var(--el-text-color-secondary);
    animation: blink 1.2s infinite;
  }

  &__feedback-summary {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-top: 4px;
  }

  &__footer {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  &__artifact-detail {
    max-height: 420px;
    overflow: auto;
    font-size: 12px;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
  }
}

@keyframes blink {
  50% {
    opacity: 0.4;
  }
}
</style>
