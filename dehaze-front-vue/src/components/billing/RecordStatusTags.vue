<!-- 计费明细行状态标签：降级 / 缓存节省 / 申诉状态 -->
<script lang="ts" setup>
import type { BillingRecordVO } from "dehaze-sdk-js";
import { computed } from "vue";

defineOptions({ name: "RecordStatusTags" });

const props = defineProps<{
  record: BillingRecordVO;
}>();

const REFUND_STATUS_TAGS: Record<
  number,
  { label: string; type: "info" | "warning" | "success" | "danger" }
> = {
  1: { label: "待审核", type: "warning" },
  2: { label: "已通过", type: "success" },
  3: { label: "已驳回", type: "danger" },
};

const refundTag = computed(() =>
  props.record.refundStatus
    ? REFUND_STATUS_TAGS[props.record.refundStatus]
    : null
);
</script>

<template>
  <div class="flex flex-wrap gap-1">
    <el-tooltip
      v-if="record.actualModel"
      :content="`已降级为 ${record.actualModel}`"
      placement="top"
    >
      <el-tag type="warning" size="small">降级</el-tag>
    </el-tooltip>
    <el-tag v-if="record.creditsSaved > 0" type="success" size="small">
      缓存省{{ record.creditsSaved }}积分
    </el-tag>
    <el-tag v-if="refundTag" :type="refundTag.type" size="small">
      {{ refundTag.label }}
    </el-tag>
    <span v-if="!record.actualModel && !record.creditsSaved && !refundTag"
      >-</span
    >
  </div>
</template>
