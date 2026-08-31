<!-- 文档处理状态标签 -->
<script lang="ts" setup>
import type { DocumentProcessingStatus } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "DocumentStatusTags" });

const props = defineProps<{
  status: DocumentProcessingStatus;
  error?: string;
}>();

const STATUS_META: Record<
  DocumentProcessingStatus,
  { label: string; type: "info" | "warning" | "success" | "danger" }
> = {
  pending: { label: "待处理", type: "info" },
  processing: { label: "处理中", type: "warning" },
  completed: { label: "已完成", type: "success" },
  failed: { label: "失败", type: "danger" },
};

const meta = computed(
  () => STATUS_META[props.status] ?? { label: "待处理", type: "info" as const }
);
</script>

<template>
  <el-tooltip
    v-if="status === 'failed' && error"
    :content="error"
    placement="top"
  >
    <el-tag :type="meta.type" size="small">{{ meta.label }}</el-tag>
  </el-tooltip>
  <el-tag v-else :type="meta.type" size="small">{{ meta.label }}</el-tag>
</template>
