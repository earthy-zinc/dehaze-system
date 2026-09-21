<!-- 消息列表：按时间正序渲染，role 分流组件，自动滚动跟随；>100 条启用轻量窗口化。
     无状态：数据 props 进、事件 emit 出（中断恢复经 resume-interrupt 交由页面层调用 store）。 -->
<script lang="ts" setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { shouldShowSuggestions } from "../vm";
import AssistantMessage from "./AssistantMessage.vue";
import InterruptCard from "./InterruptCard.vue";
import ToolMessage from "./ToolMessage.vue";
import UserMessage from "./UserMessage.vue";
import type {
  ChatArtifactVM,
  ChatAssistantMessageVM,
  ChatContextChipVM,
  ChatFeedbackVM,
  ChatInterruptVM,
  ChatMessageVM,
  ChatResumeFormVM,
  ChatScopeVM,
  ChatUserMessageVM,
} from "../types";

defineOptions({ name: "MessageList" });

const props = defineProps<{
  messages: ChatMessageVM[];
  scope: ChatScopeVM;
  streamingMessageId: number | null;
  interruptedMessageId: number | null;
  interrupts: ChatInterruptVM[];
  scrollFollowEnabled: boolean;
  /** 是否还有更早的历史消息（滚动到顶触发 reach-top 加载） */
  hasMoreHistory?: boolean;
  /** 历史消息加载中 */
  loadingMore?: boolean;
  /** 分支切换进行中（禁用切换器） */
  branchSwitching?: boolean;
}>();

const emit = defineEmits<{
  "scroll-follow-toggle": [enabled: boolean];
  "reach-top": [];
  edit: [message: ChatUserMessageVM];
  copy: [message: ChatMessageVM];
  quote: [message: ChatMessageVM];
  regenerate: [message: ChatAssistantMessageVM];
  feedback: [message: ChatAssistantMessageVM, data: ChatFeedbackVM | null];
  delete: [message: ChatMessageVM];
  speak: [message: ChatAssistantMessageVM];
  trace: [message: ChatMessageVM];
  "resume-interrupt": [messageId: number, data: ChatResumeFormVM];
  "open-artifact": [artifact: ChatArtifactVM];
  "apply-suggestion": [question: string];
  "load-artifacts": [messageId: number];
  retry: [message: ChatAssistantMessageVM];
  "switch-branch": [message: ChatAssistantMessageVM, index: number];
  "context-open": [chip: ChatContextChipVM];
}>();

const BOTTOM_THRESHOLD_PX = 40;
const TOP_THRESHOLD_PX = 40;

const bodyRef = ref<HTMLElement>();
const scrollTop = ref(0);
const viewportHeight = ref(600);

// ===== 轻量窗口化：超阈值时按滚动态切片挂载（变高消息不适用固定高度虚拟滚动，
//       故用估算高度的分片 + 占位高度方案，代价是占位高度非精确、快速滚动定位略偏） =====
const WINDOW_LIMIT = 100;
const EST_ITEM_HEIGHT = 96;
const MIN_WINDOW = 40;
const OVERSCAN = 10;

const windowed = computed(() => props.messages.length > WINDOW_LIMIT);

const startIndex = computed(() => {
  if (!windowed.value) return 0;
  if (props.scrollFollowEnabled) {
    return Math.max(0, props.messages.length - MIN_WINDOW - OVERSCAN);
  }
  const byScroll = Math.floor(scrollTop.value / EST_ITEM_HEIGHT) - OVERSCAN;
  return Math.max(0, Math.min(byScroll, props.messages.length - MIN_WINDOW));
});

const endIndex = computed(() => {
  if (!windowed.value) return props.messages.length;
  const count =
    Math.ceil(viewportHeight.value / EST_ITEM_HEIGHT) +
    OVERSCAN * 2 +
    MIN_WINDOW;
  return Math.min(props.messages.length, startIndex.value + count);
});

const visibleMessages = computed(() =>
  props.messages.slice(startIndex.value, endIndex.value)
);
const topSpacerHeight = computed(() => startIndex.value * EST_ITEM_HEIGHT);
const bottomSpacerHeight = computed(
  () => (props.messages.length - endIndex.value) * EST_ITEM_HEIGHT
);

function messageKey(message: ChatMessageVM, index: number) {
  return message.id != null
    ? `m-${message.id}`
    : `idx-${startIndex.value + index}`;
}

// ===== 中断卡：优先内联到对应被中断的消息，剩余作为尾部卡片兜底 =====
function interruptFor(message: ChatMessageVM): ChatInterruptVM | undefined {
  return message.id === props.interruptedMessageId
    ? props.interrupts[0]
    : undefined;
}

const hasMatchedInterrupt = computed(
  () =>
    props.interruptedMessageId != null &&
    props.messages.some((m) => m.id === props.interruptedMessageId)
);

const orphanInterrupts = computed(() =>
  hasMatchedInterrupt.value ? props.interrupts.slice(1) : props.interrupts
);

function resumeOrphan(data: ChatResumeFormVM) {
  emit("resume-interrupt", props.interruptedMessageId ?? 0, data);
}

// ===== 滚动跟随 =====
function scrollToBottom() {
  nextTick(() => {
    if (bodyRef.value) bodyRef.value.scrollTop = bodyRef.value.scrollHeight;
  });
}

function handleScroll() {
  const el = bodyRef.value;
  if (!el) return;
  scrollTop.value = el.scrollTop;
  if (el.clientHeight) viewportHeight.value = el.clientHeight;
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
  if (distance < BOTTOM_THRESHOLD_PX) {
    if (!props.scrollFollowEnabled) emit("scroll-follow-toggle", true);
  } else if (props.scrollFollowEnabled) {
    emit("scroll-follow-toggle", false);
  }
  if (
    el.scrollTop <= TOP_THRESHOLD_PX &&
    props.hasMoreHistory &&
    !props.loadingMore
  ) {
    emit("reach-top");
  }
}

function backToBottom() {
  scrollToBottom();
  emit("scroll-follow-toggle", true);
}

/** 消息正文长度（assistant 用 text，user/tool 用 content） */
function contentLength(message: ChatMessageVM | undefined): number {
  if (!message) return 0;
  return message.role === "assistant"
    ? message.text.length
    : message.content.length;
}

// 流式增量经 rAF 批量写入，仅监听尾部消息（id + 内容长度）即可低成本跟随；
// 以尾部身份为 key 而非总条数，避免向上加载更早历史（前置插入）误触发吸底。
watch(
  () => {
    const last = props.messages[props.messages.length - 1];
    return `${last?.id ?? "none"}:${contentLength(last)}`;
  },
  () => {
    if (props.scrollFollowEnabled) scrollToBottom();
  }
);

watch(
  () => props.scrollFollowEnabled,
  (enabled) => {
    if (enabled) scrollToBottom();
  }
);

// 前置插入补偿：向上加载更早历史（reach-top）把整列表下移，滚动容器 scrollTop 数值不变
// 会令视口内容跳变。此处记录插入前后 scrollHeight 差值并回填 scrollTop，使原先可见内容保持在
// 视口中。判定基于插入前的真实位置（不在底部才补偿）而非 scrollFollowEnabled，使用户端与
// 管理端（恒传 true）同一逻辑保位；流式追加的吸底由尾部监听单独负责，不受影响。
const firstMessageId = computed(() => props.messages[0]?.id ?? null);

watch(
  () => [props.messages.length, firstMessageId.value] as const,
  ([length, first], [prevLength, prevFirst]) => {
    const el = bodyRef.value;
    if (!el) return;
    const prepended =
      props.hasMoreHistory && length > prevLength && first !== prevFirst;
    if (!prepended) return;
    const before = el.scrollHeight;
    // 插入前视口已在底部则无需补偿（交由尾部吸底逻辑）
    if (before - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX) return;
    nextTick(() => {
      const after = bodyRef.value?.scrollHeight ?? before;
      const delta = after - before;
      if (delta > 0 && bodyRef.value) {
        bodyRef.value.scrollTop += delta;
        scrollTop.value = bodyRef.value.scrollTop;
      }
    });
  },
  { flush: "pre" }
);

onMounted(() => {
  scrollToBottom();
});

defineExpose({ scrollToBottom });
</script>

<template>
  <div class="chat-message-list">
    <div
      ref="bodyRef"
      class="chat-message-list__body"
      @scroll.passive="handleScroll"
    >
      <div v-if="loadingMore" class="chat-message-list__loading">
        正在加载历史…
      </div>
      <div
        v-if="topSpacerHeight"
        class="chat-message-list__spacer"
        :style="{ height: `${topSpacerHeight}px` }"
      />

      <template
        v-for="(message, index) in visibleMessages"
        :key="messageKey(message, index)"
      >
        <UserMessage
          v-if="message.role === 'user'"
          :message="message"
          :scope="scope"
          @edit="(m) => emit('edit', m)"
          @copy="(m) => emit('copy', m)"
          @quote="(m) => emit('quote', m)"
        />
        <AssistantMessage
          v-else-if="message.role === 'assistant'"
          :message="message"
          :scope="scope"
          :thinking="message.thinking"
          :steps="message.steps"
          :artifacts="message.artifacts"
          :memories="message.memories"
          :feedback="message.feedback"
          :suggestions="message.suggestions"
          :show-suggestions="shouldShowSuggestions(messages, message.id)"
          :interrupt="interruptFor(message)"
          :branch-switching="branchSwitching"
          @regenerate="(m) => emit('regenerate', m)"
          @quote="(m) => emit('quote', m)"
          @feedback="(m, data) => emit('feedback', m, data)"
          @delete="(m) => emit('delete', m)"
          @speak="(m) => emit('speak', m)"
          @copy="(m) => emit('copy', m)"
          @trace="(m) => emit('trace', m)"
          @open-artifact="(a) => emit('open-artifact', a)"
          @apply-suggestion="(q) => emit('apply-suggestion', q)"
          @load-artifacts="(id) => emit('load-artifacts', id)"
          @retry="(m) => emit('retry', m)"
          @resume="(payload) => emit('resume-interrupt', message.id, payload)"
          @switch-branch="(m, i) => emit('switch-branch', m, i)"
          @context-open="(chip) => emit('context-open', chip)"
        />
        <ToolMessage
          v-else
          :message="message"
          :tool-calls="message.toolCalls"
        />
      </template>

      <InterruptCard
        v-for="(interrupt, oi) in orphanInterrupts"
        :key="`interrupt-${oi}`"
        :interrupt="interrupt"
        @resume="resumeOrphan"
      />

      <div
        v-if="bottomSpacerHeight"
        class="chat-message-list__spacer"
        :style="{ height: `${bottomSpacerHeight}px` }"
      />

      <el-empty
        v-if="messages.length === 0"
        description="开始你的第一轮对话吧"
      />
    </div>

    <Transition name="fade">
      <el-button
        v-if="!scrollFollowEnabled"
        class="chat-message-list__back-bottom"
        size="small"
        circle
        @click="backToBottom"
      >
        <span class="chat-message-list__back-icon">↓</span>
      </el-button>
    </Transition>
  </div>
</template>

<style scoped lang="scss">
.chat-message-list {
  position: relative;
  height: 100%;

  /* flex 布局父容器中允许收缩到内容以下，由 __body 内部滚动，避免撑爆容器挤走输入区 */
  min-height: 0;

  &__body {
    height: 100%;
    padding: 16px;
    overflow-y: auto;
  }

  &__loading {
    padding: 4px 0;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
    text-align: center;
  }

  &__spacer {
    pointer-events: none;
  }

  &__back-bottom {
    position: absolute;
    right: 24px;
    bottom: 24px;
  }

  &__back-icon {
    font-size: 14px;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
