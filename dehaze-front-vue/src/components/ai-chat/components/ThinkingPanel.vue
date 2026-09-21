<!-- 思考过程卡：折叠头部（思考中…/已深度思考+用时）+ 多段思考正文，流式自动吸底（纯展示，props 进）。
     设计参照业界推理模型思考块惯例：正文出现前展示"思考中"实时计时并自动跟随，
     正文开始输出后自动收起（用户手动展开过则不再干预），完成后定格耗时。 -->
<script lang="ts" setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import type { ChatThinkingVM } from "../types";

defineOptions({ name: "ThinkingPanel" });

const props = defineProps<{
  state: ChatThinkingVM;
  /** 所属消息是否仍在流式生成 */
  streaming: boolean;
  /** 正文是否已开始输出（触发自动收起） */
  answerStarted: boolean;
}>();

const open = ref(true);
const userToggled = ref(false);

// 正文出现自动收起思考区；用户手动操作过则不再干预
watch(
  () => props.answerStarted,
  (started) => {
    if (started && !userToggled.value) open.value = false;
  }
);

function toggle() {
  userToggled.value = true;
  open.value = !open.value;
}

// ===== 计时：思考中跟随前端时钟逐秒推进，结束定格 =====
// 多段思考（思考→工具→再思考）时首段 stop 会写入 endAt，故以"存在未闭合段"判定进行中
const timing = computed(
  () => props.streaming && props.state.segments.some((seg) => !seg.closed)
);
const elapsed = ref(0);
let timer: number | null = null;

watch(
  timing,
  (active) => {
    if (timer != null) {
      window.clearInterval(timer);
      timer = null;
    }
    if (active) {
      elapsed.value = Date.now() - props.state.startAt;
      timer = window.setInterval(() => {
        elapsed.value = Date.now() - props.state.startAt;
      }, 1000);
    } else if (props.state.endAt != null) {
      elapsed.value = props.state.endAt - props.state.startAt;
    }
  },
  { immediate: true }
);
onBeforeUnmount(() => {
  if (timer != null) window.clearInterval(timer);
});

function formatDuration(ms: number): string {
  const total = Math.floor(ms / 1000);
  if (total < 60) return `${total} 秒`;
  return `${Math.floor(total / 60)} 分 ${total % 60} 秒`;
}

// 不足 1 秒不展示括号，避免闪出「(0 秒)」
const title = computed(() => {
  if (timing.value) {
    return elapsed.value >= 1000
      ? `思考中…（${formatDuration(elapsed.value)}）`
      : "思考中…";
  }
  if (
    props.state.endAt != null &&
    props.state.endAt - props.state.startAt >= 1000
  ) {
    return `已深度思考（用时 ${formatDuration(props.state.endAt - props.state.startAt)}）`;
  }
  return "已深度思考";
});

// ===== 流式自动吸底：用户上滑后停止跟随 =====
const contentRef = ref<HTMLElement | null>(null);
const stickBottom = ref(true);

function onUserScroll() {
  const el = contentRef.value;
  if (!el) return;
  stickBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight <= 20;
}

const visibleSegments = computed(() =>
  props.state.segments.filter((seg) => seg.text)
);

const contentSignature = computed(() =>
  visibleSegments.value.map((seg) => seg.text.length).join(",")
);

watch(contentSignature, async () => {
  if (!open.value || !stickBottom.value) return;
  await nextTick();
  const el = contentRef.value;
  if (el) el.scrollTop = el.scrollHeight;
});
</script>

<template>
  <div v-if="visibleSegments.length" class="thinking-panel">
    <button type="button" class="thinking-panel__header" @click="toggle">
      <span
        class="thinking-panel__status"
        :class="timing ? 'is-loading' : 'is-done'"
        >{{ timing ? "◌" : "✓" }}</span
      >
      <span class="thinking-panel__title" :class="{ 'is-active': timing }">
        {{ title }}
      </span>
      <span class="thinking-panel__arrow" :class="{ 'is-open': open }">▾</span>
    </button>
    <div
      v-if="open"
      ref="contentRef"
      class="thinking-panel__content"
      @scroll.passive="onUserScroll"
    >
      <template v-for="(seg, i) in visibleSegments" :key="i">
        <div v-if="i > 0" class="thinking-panel__divider"></div>
        <div class="thinking-panel__body">
          <MarkdownRenderer :content="seg.text" />
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped lang="scss">
.thinking-panel {
  max-width: 92%;
  margin-bottom: 8px;

  &__header {
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 0;
    font-size: 13px;
    color: var(--el-text-color-secondary);
    cursor: pointer;
    background: none;
    border: none;

    &:hover .thinking-panel__title {
      color: var(--el-text-color-primary);
    }
  }

  &__status {
    font-size: 12px;

    &.is-loading {
      color: var(--el-color-primary);
      animation: spin 1.2s linear infinite;
    }

    &.is-done {
      color: var(--el-color-success);
    }
  }

  &__title.is-active {
    color: var(--el-color-primary);
  }

  &__arrow {
    color: var(--el-text-color-placeholder);
    transition: transform 0.2s;

    &.is-open {
      transform: rotate(180deg);
    }
  }

  &__content {
    max-height: 300px;
    padding: 8px 12px;
    margin-top: 4px;
    overflow: auto;
    font-size: 13px;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }

  &__body :deep(p) {
    margin: 0 0 4px;
  }

  &__divider {
    height: 1px;
    margin: 8px 0;
    background-color: var(--el-border-color-lighter);
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
