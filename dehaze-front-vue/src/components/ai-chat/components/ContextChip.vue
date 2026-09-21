<!-- 上下文构成标签：类别 + 占比条 + 可查看详情（emit open；纯展示） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatContextChipVM } from "../types";

defineOptions({ name: "ContextChip" });

const props = defineProps<{
  chip: ChatContextChipVM;
}>();

const emit = defineEmits<{
  open: [chip: ChatContextChipVM];
}>();

const kindLabels: Record<ChatContextChipVM["kind"], string> = {
  system: "系统提示",
  history: "历史",
  memory: "记忆",
  retrieval: "检索",
  tools: "工具",
};

const label = computed(() => props.chip.label || kindLabels[props.chip.kind]);

const ratioPercent = computed(() => {
  const ratio = props.chip.ratio;
  if (ratio == null) return null;
  return Math.max(0, Math.min(100, ratio));
});
</script>

<template>
  <div
    class="context-chip"
    :class="{ 'context-chip--clickable': !!chip.detail }"
    @click="chip.detail && emit('open', chip)"
  >
    <span class="context-chip__label">{{ label }}</span>
    <span v-if="ratioPercent != null" class="context-chip__ratio">
      {{ ratioPercent.toFixed(0) }}%
    </span>
    <span
      v-if="ratioPercent != null"
      class="context-chip__bar"
      :style="{ width: `${ratioPercent}%` }"
    />
  </div>
</template>

<style scoped lang="scss">
.context-chip {
  position: relative;
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 2px 8px;
  overflow: hidden;
  font-size: 12px;
  color: var(--el-text-color-regular);
  background-color: var(--el-fill-color-light);
  border-radius: 12px;

  &--clickable {
    cursor: pointer;

    &:hover {
      color: var(--el-color-primary);
    }
  }

  &__label {
    position: relative;
    z-index: 1;
  }

  &__ratio {
    position: relative;
    z-index: 1;
    color: var(--el-text-color-secondary);
  }

  &__bar {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background-color: var(--el-color-primary-light-7);
  }
}
</style>
