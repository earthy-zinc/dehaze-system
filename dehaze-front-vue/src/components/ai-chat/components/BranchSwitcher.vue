<!-- 消息分支切换器（< 2/3 >）：total<2 不渲染；边界禁用；change 上抛目标序号（纯展示） -->
<script lang="ts" setup>
import { computed } from "vue";

defineOptions({ name: "BranchSwitcher" });

const props = defineProps<{
  total: number;
  /** 当前分支序号（从 1 计） */
  current: number;
  /** 切换进行中：禁用交互（防重复提交，等待后端切换完成） */
  disabled?: boolean;
}>();

const emit = defineEmits<{
  change: [index: number];
}>();

const canPrev = computed(() => !props.disabled && props.current > 1);
const canNext = computed(() => !props.disabled && props.current < props.total);

function prev() {
  if (canPrev.value) emit("change", props.current - 1);
}

function next() {
  if (canNext.value) emit("change", props.current + 1);
}
</script>

<template>
  <div v-if="total > 1" class="branch-switcher">
    <el-button
      link
      size="small"
      :disabled="!canPrev"
      class="branch-switcher__prev"
      @click="prev"
      >&lt;</el-button
    >
    <span class="branch-switcher__index">{{ current }}/{{ total }}</span>
    <el-button
      link
      size="small"
      :disabled="!canNext"
      class="branch-switcher__next"
      @click="next"
      >&gt;</el-button
    >
  </div>
</template>

<style scoped lang="scss">
.branch-switcher {
  display: inline-flex;
  gap: 4px;
  align-items: center;
  font-size: 12px;

  &__index {
    min-width: 32px;
    color: var(--el-text-color-secondary);
    text-align: center;
  }
}
</style>
