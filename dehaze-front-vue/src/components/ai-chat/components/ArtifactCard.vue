<!-- 产物卡片：图片缩略图/指标报告/算法推荐/文件引用（纯展示，props 进、open 事件出） -->
<script lang="ts" setup>
import { computed } from "vue";
import type { ChatArtifactVM } from "../types";

defineOptions({ name: "ArtifactCard" });

const props = defineProps<{
  artifact: ChatArtifactVM;
}>();

const emit = defineEmits<{
  open: [artifact: ChatArtifactVM];
}>();

const typeMeta = computed(() => {
  switch (props.artifact.type) {
    case "image_result":
      return { label: "图片结果" };
    case "metric_report":
      return { label: "指标报告" };
    case "algorithm_recommend":
      return { label: "算法推荐" };
    case "file_ref":
      return { label: "文件引用" };
    default:
      return { label: "产物" };
  }
});

const summaryText = computed(() => {
  const summary = props.artifact.summary;
  if (summary == null) return "";
  const text = typeof summary === "string" ? summary : JSON.stringify(summary);
  return text.length > 120 ? `${text.slice(0, 120)}…` : text;
});
</script>

<template>
  <div class="artifact-card" @click="emit('open', artifact)">
    <div class="artifact-card__header">
      <el-tag size="small">{{ typeMeta.label }}</el-tag>
      <el-tag v-if="artifact.invalid" size="small" type="danger">已失效</el-tag>
    </div>
    <div v-if="summaryText" class="artifact-card__summary">
      {{ summaryText }}
    </div>
  </div>
</template>

<style scoped lang="scss">
.artifact-card {
  max-width: 92%;
  padding: 10px 12px;
  margin-bottom: 8px;
  cursor: pointer;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;

  &:hover {
    border-color: var(--el-color-primary-light-5);
  }

  &__header {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  &__summary {
    margin-top: 6px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
    word-break: break-all;
  }
}
</style>
