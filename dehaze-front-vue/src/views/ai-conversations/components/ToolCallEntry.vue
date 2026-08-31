<!-- 工具调用条目：工具名/参数/结果摘要/首Token耗时，默认折叠 -->
<script lang="ts" setup>
import { computed } from "vue";

defineOptions({ name: "ToolCallEntry" });

const props = defineProps<{
  name: string;
  args?: unknown;
  result?: string;
  latencyMs?: number;
}>();

const collapseTitle = computed(() =>
  props.latencyMs != null
    ? `工具调用：${props.name} · 首Token ${props.latencyMs}ms`
    : `工具调用：${props.name}`
);

function formatValue(value: unknown) {
  if (value == null) return "";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
</script>

<template>
  <el-collapse class="tool-call-entry">
    <el-collapse-item :title="collapseTitle" name="tool">
      <div v-if="args != null" class="tool-call-entry__block">
        <div class="tool-call-entry__label">参数</div>
        <pre>{{ formatValue(args) }}</pre>
      </div>
      <div v-if="result" class="tool-call-entry__block">
        <div class="tool-call-entry__label">结果摘要</div>
        <pre>{{ result }}</pre>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped lang="scss">
.tool-call-entry {
  border: none;

  :deep(.el-collapse-item__header) {
    height: 32px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  &__block {
    & + & {
      margin-top: 8px;
    }
  }

  &__label {
    margin-bottom: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }

  pre {
    max-height: 160px;
    padding: 6px 8px;
    margin: 0;
    overflow: auto;
    font-size: 12px;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    background-color: var(--el-fill-color-light);
    border-radius: 6px;
  }
}
</style>
