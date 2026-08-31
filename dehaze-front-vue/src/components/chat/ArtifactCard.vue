<!-- 产物卡片：图片缩略图/指标报告/算法推荐/文件引用 -->
<script lang="ts" setup>
import type { ArtifactVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "ArtifactCard" });

const props = defineProps<{
  artifact: ArtifactVO;
  detailUrl?: string;
}>();

const emit = defineEmits<{
  open: [artifact: ArtifactVO];
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
      <el-tag v-if="artifact.isInvalid === 1" size="small" type="danger"
        >已失效</el-tag
      >
      <a
        v-if="detailUrl"
        :href="detailUrl"
        target="_blank"
        rel="noopener noreferrer"
        class="artifact-card__link"
        @click.stop
      >
        查看详情
      </a>
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

  &__link {
    margin-left: auto;
    font-size: 12px;
  }

  &__summary {
    margin-top: 6px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
    word-break: break-all;
  }
}
</style>
