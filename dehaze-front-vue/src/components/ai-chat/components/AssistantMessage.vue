<!-- 助手消息：聚合 Markdown/思考过程/推理链/产物/记忆引用/反馈/推荐问题/操作栏（纯展示）。
     数据全部 props 进；产物详情/中断恢复等副作用交由页面层，组件仅 emit。 -->
<script lang="ts" setup>
import { computed, onMounted, ref, watch } from "vue";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import { filterToolSteps } from "../vm/steps";
import ArtifactCard from "./ArtifactCard.vue";
import BranchSwitcher from "./BranchSwitcher.vue";
import FeedbackBar from "./FeedbackBar.vue";
import InterruptCard from "./InterruptCard.vue";
import MemoryReferenceList from "./MemoryReferenceList.vue";
import MessageActionBar from "./MessageActionBar.vue";
import PlanPanel from "./PlanPanel.vue";
import ProcessPanel from "./ProcessPanel.vue";
import SuggestionList from "./SuggestionList.vue";
import ThinkingPanel from "./ThinkingPanel.vue";
import ThoughtChain from "./ThoughtChain.vue";
import type {
  ChatArtifactVM,
  ChatAssistantMessageVM,
  ChatContextChipVM,
  ChatFeedbackVM,
  ChatInterruptVM,
  ChatMemoryVM,
  ChatResumeFormVM,
  ChatScopeVM,
  ChatThinkingVM,
  ChatThoughtStepVM,
} from "../types";

defineOptions({ name: "AssistantMessage" });

const props = defineProps<{
  message: ChatAssistantMessageVM;
  scope: ChatScopeVM;
  thinking: ChatThinkingVM | null;
  steps: ChatThoughtStepVM[];
  artifacts: ChatArtifactVM[];
  memories: ChatMemoryVM[];
  feedback: ChatFeedbackVM | null;
  suggestions: string[];
  showSuggestions: boolean;
  interrupt?: ChatInterruptVM;
  /** 分支切换进行中：禁用本消息切换器 */
  branchSwitching?: boolean;
}>();

const emit = defineEmits<{
  regenerate: [message: ChatAssistantMessageVM];
  quote: [message: ChatAssistantMessageVM];
  feedback: [message: ChatAssistantMessageVM, data: ChatFeedbackVM | null];
  delete: [message: ChatAssistantMessageVM];
  speak: [message: ChatAssistantMessageVM];
  copy: [message: ChatAssistantMessageVM];
  trace: [message: ChatAssistantMessageVM];
  "open-artifact": [artifact: ChatArtifactVM];
  "apply-suggestion": [question: string];
  "load-artifacts": [messageId: number];
  retry: [message: ChatAssistantMessageVM];
  resume: [payload: ChatResumeFormVM];
  /** 分支切换：交由宿主绑定层切换当前展示分支 */
  "switch-branch": [message: ChatAssistantMessageVM, index: number];
  /** 上下文构成项详情查看（用户端无可下钻原始来源，由页面层展示人类可读说明） */
  "context-open": [chip: ChatContextChipVM];
}>();

// 推理链只承载工具步骤：纯思考记录由 ThinkingPanel 展示
const toolSteps = computed(() => filterToolSteps(props.steps));

const isStreaming = computed(() => props.message.status === "streaming");
const isFailed = computed(() => props.message.status === "failed");
const isCanceled = computed(() => props.message.status === "canceled");

const feedbackEditing = ref(false);

function maybeLoadArtifacts() {
  if (props.message.id > 0 && props.message.status !== "streaming") {
    emit("load-artifacts", props.message.id);
  }
}

onMounted(maybeLoadArtifacts);
// 流式完成（streaming → 终态）时补拉产物
watch(() => props.message.status, maybeLoadArtifacts);

function handleFeedbackSubmit(data: ChatFeedbackVM | null) {
  feedbackEditing.value = false;
  emit("feedback", props.message, data);
}

function handleFeedbackCancel() {
  feedbackEditing.value = false;
  // 已有反馈时 cancel 语义为撤销反馈
  if (props.feedback) emit("feedback", props.message, null);
}
</script>

<template>
  <div class="message-block assistant-message">
    <!-- 分支切换器（同一分叉点兄弟消息前端派生）：切换查看不同重新生成/编辑分支 -->
    <BranchSwitcher
      v-if="message.branch"
      :total="message.branch.total"
      :current="message.branch.current"
      :disabled="branchSwitching"
      @change="(index) => emit('switch-branch', message, index)"
    />

    <!-- 计划面板（Plan-and-Execute）：计划生成/更新/重规划可见 -->
    <PlanPanel v-if="message.plan" :plan="message.plan" />

    <!-- 推理模型：思考过程折叠卡（思考中实时计时，正文出现自动收起），正文在下 -->
    <ThinkingPanel
      v-if="thinking"
      :state="thinking"
      :streaming="isStreaming"
      :answer-started="!!message.text"
    />

    <ThoughtChain
      v-if="toolSteps.length > 0"
      :thoughts="toolSteps"
      :sub-agents="message.usage?.subAgents"
    />

    <div class="assistant-message__bubble">
      <template v-if="isStreaming && !message.text">
        <span class="assistant-message__typing">正在思考…</span>
      </template>
      <MarkdownRenderer v-else-if="message.text" :content="message.text" />
      <el-alert
        v-if="isFailed"
        type="error"
        :closable="false"
        :title="message.error || '回复生成失败'"
      />
      <el-tag v-if="isCanceled" type="info" size="small">已停止生成</el-tag>
      <el-button
        v-if="isFailed"
        link
        size="small"
        @click="emit('retry', message)"
        >重试</el-button
      >
    </div>

    <!-- 过程透明（用户端"查看过程"）：默认折叠，展开看步骤时间线/摘要/上下文构成，不透出 raw 报文 -->
    <ProcessPanel
      v-if="message.process"
      :steps="message.process.steps"
      :summary="message.process.summary"
      :chips="message.process.chips"
      @context-open="(chip) => emit('context-open', chip)"
    />

    <ArtifactCard
      v-for="artifact in artifacts"
      :key="artifact.id"
      :artifact="artifact"
      @open="(a) => emit('open-artifact', a)"
    />

    <MemoryReferenceList v-if="memories.length > 0" :memories="memories" />

    <InterruptCard
      v-if="interrupt"
      :interrupt="interrupt"
      @resume="(payload) => emit('resume', payload)"
    />

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
      :questions="suggestions"
      @apply="(question) => emit('apply-suggestion', question)"
    />

    <div class="assistant-message__footer">
      <MessageActionBar
        :message="message"
        :scope="scope"
        @copy="emit('copy', message)"
        @quote="emit('quote', message)"
        @regenerate="emit('regenerate', message)"
        @speak="emit('speak', message)"
        @delete="emit('delete', message)"
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
  </div>
</template>

<style scoped lang="scss">
.assistant-message {
  margin-bottom: 16px;

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
}

@keyframes blink {
  50% {
    opacity: 0.4;
  }
}
</style>
