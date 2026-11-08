<!-- 消息列表：按时间正序渲染，role 分流组件，自动滚动跟随；>100 条启用轻量窗口化。
     无状态：数据 props 进、事件 emit 出；内容动作统一上抛 message-action（判别联合）。 -->
<script lang="ts" setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";
import type { Artifact, Feedback, Message } from "clarity-sdk-js";
import { textOfBlocks } from "../model/blocks";
import type { ChatBranchVM, ChatMessageActionVM, ChatScopeVM } from "../types";
import AssistantMessage from "./AssistantMessage.vue";
import UserMessage from "./UserMessage.vue";

defineOptions({ name: "MessageList" });

const props = defineProps<{
  messages: Message[];
  /** 宿主补充数据（按消息 id 索引）：产物/反馈/分支序号，消息实体本身不携带 */
  artifacts?: Record<number, Artifact[]>;
  feedback?: Record<number, Feedback | null>;
  branches?: Record<number, ChatBranchVM>;
  scope: ChatScopeVM;
  scrollFollowEnabled: boolean;
  /** 是否还有更早的历史消息（滚动到顶触发 reach-top 加载） */
  hasMoreHistory?: boolean;
  /** 历史消息加载中 */
  loadingMore?: boolean;
  /** 历史加载失败（initial 首屏 / more 上翻）：行内提示并给出重试入口 */
  error?: { kind: "initial" | "more"; message: string } | null;
  /** 错误重试进行中（禁用重试按钮） */
  retrying?: boolean;
  /** 分支切换禁用（切换进行中或流式/中断中） */
  branchDisabled?: boolean;
}>();

const emit = defineEmits<{
  "scroll-follow-toggle": [enabled: boolean];
  "reach-top": [];
  /** 历史加载失败后重试 */
  retry: [];
  /** 单一消息动作（复制/编辑/引用/重新生成/重试/朗读/删除/反馈/恢复中断/采用推荐问题） */
  "message-action": [action: ChatMessageActionVM];
  "open-artifact": [artifact: Artifact];
  "load-artifacts": [messageId: number];
  "switch-branch": [message: Message, index: number];
  trace: [message: Message];
}>();

/** 推荐追问仅挂在最后一条 assistant 消息之后 */
const lastAssistantId = computed<number | null>(() => {
  for (let i = props.messages.length - 1; i >= 0; i--) {
    const message = props.messages[i];
    if (message.role === "assistant") return message.id;
  }
  return null;
});

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

function messageKey(message: Message, index: number) {
  return message.id != null
    ? `m-${message.id}`
    : `idx-${startIndex.value + index}`;
}

// ===== 中断卡由消息实体的 pendingInterrupt 内联渲染（无侧表、无兜底散卡） =====

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
    !props.loadingMore &&
    !props.error
  ) {
    emit("reach-top");
  }
}

function backToBottom() {
  scrollToBottom();
  emit("scroll-follow-toggle", true);
}

/** 消息正文长度（text 块拼接，块增量推进时随之变化） */
function contentLength(message: Message | undefined): number {
  return message ? textOfBlocks(message.blocks).length : 0;
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
      <div v-if="error" class="chat-message-list__error">
        <span class="chat-message-list__error-text">
          历史消息加载失败：{{ error.message }}
        </span>
        <el-button
          link
          size="small"
          type="primary"
          class="chat-message-list__retry"
          :disabled="retrying"
          @click="emit('retry')"
          >重试</el-button
        >
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
          @action="(action) => emit('message-action', action)"
        />
        <AssistantMessage
          v-else-if="message.role === 'assistant'"
          :message="message"
          :artifacts="artifacts?.[message.id]"
          :feedback="feedback?.[message.id] ?? null"
          :branch="branches?.[message.id]"
          :can-delete="message.id === lastAssistantId"
          :scope="scope"
          :show-suggestions="message.id === lastAssistantId"
          :branch-disabled="branchDisabled"
          @action="(action) => emit('message-action', action)"
          @open-artifact="(a) => emit('open-artifact', a)"
          @load-artifacts="(id) => emit('load-artifacts', id)"
          @switch-branch="(m, i) => emit('switch-branch', m, i)"
          @trace="(m) => emit('trace', m)"
        />
      </template>

      <div
        v-if="bottomSpacerHeight"
        class="chat-message-list__spacer"
        :style="{ height: `${bottomSpacerHeight}px` }"
      />

      <el-empty
        v-if="messages.length === 0 && !error"
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

  &__error {
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: center;
    padding: 4px 0;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--el-color-danger);
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
