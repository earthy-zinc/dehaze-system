<!-- 助手消息：聚合 Markdown/思考过程/推理步骤/产物/反馈/推荐问题/操作栏（纯展示）。
     消息内容全部来自契约 Message：正文/思考/步骤由 blocks 派生，计划/中断/用量/推荐问题/
     记忆引用直接读契约字段；产物/反馈/分支由宿主按消息 id 注入。
     副作用（产物详情、中断恢复）由宿主处理，组件仅上抛单一 action。 -->
<script lang="ts" setup>
import { computed, onMounted, ref, watch } from "vue";
import type { Artifact, Feedback, FeedbackForm, Message } from "clarity-sdk-js";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import {
  hasUsage,
  stepsOfBlocks,
  textOfBlocks,
  thinkingOfBlocks,
} from "../model/blocks";
import type {
  ChatBranchVM,
  ChatMessageActionVM,
  ChatResumeFormVM,
  ChatScopeVM,
} from "../types";
import ArtifactCard from "./ArtifactCard.vue";
import BranchSwitcher from "./BranchSwitcher.vue";
import FeedbackBar from "./FeedbackBar.vue";
import InterruptCard from "./InterruptCard.vue";
import MessageActionBar from "./MessageActionBar.vue";
import PlanPanel from "./PlanPanel.vue";
import SuggestionList from "./SuggestionList.vue";
import ThinkingPanel from "./ThinkingPanel.vue";
import ThoughtChain from "./ThoughtChain.vue";

defineOptions({ name: "AssistantMessage" });

/** 无产物时复用同一空数组，避免每次渲染产生新引用 */
const EMPTY_ARTIFACTS: Artifact[] = [];

const props = withDefaults(
  defineProps<{
    message: Message;
    /** 宿主补充数据（消息实体之外，按 id 注入） */
    artifacts?: Artifact[];
    feedback?: Feedback | null;
    /** 同一分叉点的分支序号（兄弟数大于 1 时由宿主给出） */
    branch?: ChatBranchVM;
    /** 是否为激活链最后一条助手消息：只有叶子回复可删除 */
    canDelete?: boolean;
    scope: ChatScopeVM;
    showSuggestions: boolean;
    /** 分支切换禁用（切换进行中，或流式/中断期间禁止切换） */
    branchDisabled?: boolean;
  }>(),
  { canDelete: true }
);

const emit = defineEmits<{
  /** 单一内容动作（copy/quote/regenerate/retry/speak/delete/feedback/resume-interrupt/apply-suggestion） */
  action: [action: ChatMessageActionVM];
  "open-artifact": [artifact: Artifact];
  "load-artifacts": [messageId: number];
  /** 分支切换：交由宿主绑定层切换当前展示分支 */
  "switch-branch": [message: Message, index: number];
  /** 管理端链路下钻（跳转可观测性时间线） */
  trace: [message: Message];
}>();

function emitAction(action: ChatMessageActionVM) {
  emit("action", action);
}

// 正文 / 思考 / 推理步骤：组件按 blocks 现算，块变化即重算，不依赖宿主投影
const text = computed(() => textOfBlocks(props.message.blocks));
const thinking = computed(() =>
  thinkingOfBlocks(
    props.message.blocks,
    Date.parse(props.message.createTime) || 0
  )
);
const steps = computed(() => stepsOfBlocks(props.message.blocks));
const artifacts = computed(() => props.artifacts ?? EMPTY_ARTIFACTS);
const branch = computed(() => props.branch);
const feedback = computed(() => props.feedback ?? null);

const isStreaming = computed(() => props.message.status === "running");
const isFailed = computed(() => props.message.status === "failed");
const isCanceled = computed(() => props.message.status === "canceled");

/** 逐条消息的用量：有数据且不在流式中时展示（契约 usage 是唯一来源） */
const usageText = computed(() => {
  const usage = props.message.usage;
  if (isStreaming.value || !hasUsage(usage)) return "";
  const cached =
    usage.cachedInputTokens > 0 ? `（缓存 ${usage.cachedInputTokens}）` : "";
  return `输入 ${usage.inputTokens} / 输出 ${usage.outputTokens} tokens${cached} · 积分 ${usage.credits}`;
});

/** 记忆引用：只报条数，不伪造上下文构成与占比 */
const memoryText = computed(() =>
  props.message.memoryIds.length > 0
    ? `引用记忆 ${props.message.memoryIds.length} 条`
    : ""
);

const feedbackEditing = ref(false);

function maybeLoadArtifacts() {
  if (props.message.id > 0 && props.message.status !== "running") {
    emit("load-artifacts", props.message.id);
  }
}

onMounted(maybeLoadArtifacts);
// 流式完成（running → 终态）时补拉产物
watch(() => props.message.status, maybeLoadArtifacts);

function handleFeedbackSubmit(data: FeedbackForm | null) {
  feedbackEditing.value = false;
  emitAction({ type: "feedback", messageId: props.message.id, data });
}

function handleFeedbackCancel() {
  feedbackEditing.value = false;
  // 已有反馈时 cancel 语义为撤销反馈
  if (feedback.value)
    emitAction({ type: "feedback", messageId: props.message.id, data: null });
}

function handleResume(payload: ChatResumeFormVM) {
  emitAction({
    type: "resume-interrupt",
    messageId: props.message.id,
    data: payload,
  });
}
</script>

<template>
  <div class="message-block assistant-message">
    <!-- 分支切换器（同一分叉点兄弟消息前端派生）：切换查看不同重新生成/编辑分支 -->
    <BranchSwitcher
      v-if="branch"
      :total="branch.total"
      :current="branch.current"
      :disabled="branchDisabled"
      @change="(index) => emit('switch-branch', message, index)"
    />

    <!-- 计划面板（Plan-and-Execute）：计划生成/更新/重规划可见 -->
    <PlanPanel v-if="message.plan" :plan="message.plan" />

    <!-- 推理模型：思考过程折叠卡（思考中实时计时，正文出现自动收起），正文在下 -->
    <ThinkingPanel
      v-if="thinking"
      :state="thinking"
      :streaming="isStreaming"
      :answer-started="!!text"
    />

    <ThoughtChain
      v-if="steps.length > 0"
      :steps="steps"
      :sub-agents="message.usage.subAgents ?? undefined"
    />

    <div class="assistant-message__bubble">
      <template v-if="isStreaming && !text">
        <span class="assistant-message__typing">正在思考…</span>
      </template>
      <MarkdownRenderer
        v-else-if="text"
        :content="text"
        :streaming="isStreaming"
      />
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
        @click="emitAction({ type: 'retry', messageId: message.id })"
        >重试</el-button
      >
    </div>

    <ArtifactCard
      v-for="artifact in artifacts"
      :key="artifact.id"
      :artifact="artifact"
      @open="(a) => emit('open-artifact', a)"
    />

    <InterruptCard
      v-if="message.pendingInterrupt"
      :interrupt="message.pendingInterrupt"
      @resume="handleResume"
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
      v-if="showSuggestions && !isStreaming && message.suggestions.length > 0"
      :questions="message.suggestions"
      @apply="(question) => emitAction({ type: 'apply-suggestion', question })"
    />

    <div class="assistant-message__footer">
      <span v-if="usageText" class="assistant-message__meta">{{
        usageText
      }}</span>
      <span v-if="memoryText" class="assistant-message__meta">{{
        memoryText
      }}</span>
      <MessageActionBar
        :message="message"
        :can-delete="canDelete"
        :scope="scope"
        @action="emitAction"
        @feedback="feedbackEditing = true"
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

  &__meta {
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

@keyframes blink {
  50% {
    opacity: 0.4;
  }
}
</style>
