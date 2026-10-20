<!-- 过程面板：默认折叠，展开后展示"AI 执行过程"（上下文构成 + 步骤时间线 + 摘要）；不透出 raw 报文 -->
<script lang="ts" setup>
import { computed, ref } from "vue";
import ContextChip from "./ContextChip.vue";
import StepTimeline from "./StepTimeline.vue";
import TraceSummary from "./TraceSummary.vue";
import type {
  ChatContextChipVM,
  ChatTraceStepVM,
  ChatTraceSummaryVM,
} from "../types";

defineOptions({ name: "ProcessPanel" });

const props = defineProps<{
  steps: ChatTraceStepVM[];
  summary: ChatTraceSummaryVM;
  chips?: ChatContextChipVM[];
  /** 初始展开态（默认折叠） */
  defaultOpen?: boolean;
}>();

const emit = defineEmits<{
  "context-open": [chip: ChatContextChipVM];
  "step-select": [step: ChatTraceStepVM];
}>();

const open = ref(props.defaultOpen ?? false);

const hasFailed = computed(
  () =>
    (props.summary.failedCount ?? 0) > 0 ||
    props.steps.some((step) => step.status === "failed")
);

const contextChips = computed(() => props.chips ?? []);

function toggle() {
  open.value = !open.value;
}
</script>

<template>
  <div class="process-panel" :class="{ 'is-open': open }">
    <button type="button" class="process-panel__header" @click="toggle">
      <span class="process-panel__arrow" :class="{ 'is-open': open }">▾</span>
      <span class="process-panel__title">查看过程</span>
      <el-tag v-if="hasFailed" type="danger" size="small">存在失败步骤</el-tag>
      <span class="process-panel__meta">{{ summary.stepCount }} 步</span>
    </button>

    <div v-if="open" class="process-panel__body">
      <div v-if="contextChips.length" class="process-panel__context">
        <ContextChip
          v-for="(chip, index) in contextChips"
          :key="`${chip.kind}-${index}`"
          :chip="chip"
          @open="(c) => emit('context-open', c)"
        />
      </div>

      <TraceSummary :summary="summary" class="process-panel__summary" />

      <StepTimeline
        :steps="steps"
        class="process-panel__steps"
        @select="(step) => emit('step-select', step)"
      />

      <el-empty
        v-if="steps.length === 0"
        :image-size="40"
        description="暂无过程记录"
      />
    </div>
  </div>
</template>

<style scoped lang="scss">
.process-panel {
  max-width: 92%;
  margin-bottom: 8px;

  &__header {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 0;
    font-size: 13px;
    color: var(--el-text-color-secondary);
    cursor: pointer;
    background: none;
    border: none;

    &:hover .process-panel__title {
      color: var(--el-text-color-primary);
    }
  }

  &__arrow {
    transition: transform 0.2s;

    &.is-open {
      transform: rotate(180deg);
    }
  }

  &__meta {
    font-size: 12px;
    color: var(--el-text-color-placeholder);
  }

  &__body {
    padding: 8px 12px;
    margin-top: 4px;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }

  &__context {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  &__summary {
    margin-bottom: 8px;
  }
}
</style>
