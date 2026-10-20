<!-- 异常监控：异常规则筛选 + 时间范围 + 分页表格 -->
<template>
  <div>
    <div class="flex items-center gap-2 mb-2 flex-wrap">
      <el-select
        v-model="billingStore.anomalyFilter.anomalyType"
        clearable
        placeholder="异常类型"
        style="width: 180px"
        @change="handleQuery"
      >
        <el-option
          v-for="option in ANOMALY_TYPE_OPTIONS"
          :key="option.value"
          :label="option.label"
          :value="option.value"
        />
      </el-select>
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        value-format="YYYY-MM-DD"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        @change="handleQuery"
      />
    </div>

    <el-table
      v-loading="billingStore.anomalyLoading"
      :data="billingStore.anomalies"
      size="small"
    >
      <el-table-column label="用户ID" prop="userId" width="90" align="center" />
      <el-table-column label="异常类型" width="130" align="center">
        <template #default="{ row }">
          <el-tag type="warning" size="small">{{
            anomalyTypeLabel(row.anomalyType)
          }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="详情"
        prop="detail"
        min-width="220"
        show-overflow-tooltip
      />
      <el-table-column label="状态" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="statusTag(row.status).type" size="small">
            {{ statusTag(row.status).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="触发时间" prop="triggerAt" width="160" />
    </el-table>

    <pagination
      v-if="billingStore.anomalyTotal > billingStore.anomalyPageSize"
      v-model:limit="billingStore.anomalyPageSize"
      v-model:page="billingStore.anomalyPageNum"
      v-model:total="billingStore.anomalyTotal"
      @pagination="billingStore.fetchAnomalies()"
    />
  </div>
</template>

<script lang="ts" setup>
import { useAdminBillingStore } from "@/store/modules/adminBilling";

defineOptions({ name: "AnomalyMonitor" });

const billingStore = useAdminBillingStore();

const dateRange = ref<[string, string]>(["", ""]);

const ANOMALY_TYPE_OPTIONS: { value: string; label: string }[] = [
  { value: "single_high", label: "单次超高" },
  { value: "burst", label: "突发峰值" },
  { value: "consecutive_quota_fail", label: "连续配额不足" },
  { value: "empty_high_output", label: "空回复高耗" },
];

function anomalyTypeLabel(type: string) {
  return (
    ANOMALY_TYPE_OPTIONS.find((option) => option.value === type)?.label ?? type
  );
}

function statusTag(status: number) {
  switch (status) {
    case 1:
      return { label: "已处理", type: "success" as const };
    case 2:
      return { label: "已忽略", type: "info" as const };
    default:
      return { label: "待处理", type: "warning" as const };
  }
}

function handleQuery() {
  billingStore.anomalyFilter.dateStart = dateRange.value?.[0];
  billingStore.anomalyFilter.dateEnd = dateRange.value?.[1];
  billingStore.anomalyPageNum = 1;
  billingStore.fetchAnomalies();
}

onMounted(() => {
  billingStore.fetchAnomalies();
});
</script>
