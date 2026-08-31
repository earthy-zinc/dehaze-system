<!-- 异常监控：异常类型筛选 + 时间范围 + 分页表格 -->
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
        <el-option label="异常计费" value="anomalous" />
        <el-option label="人工调整" value="manual" />
        <el-option label="自动补偿" value="auto_compensated" />
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
      <el-table-column label="用户" min-width="120">
        <template #default="{ row }">
          {{ row.username ?? row.userId }}
        </template>
      </el-table-column>
      <el-table-column label="异常类型" width="120" align="center">
        <template #default="{ row }">
          <el-tag type="warning" size="small">{{
            anomalyTypeLabel(row.anomalyType)
          }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="理论成本（积分）"
        prop="costCredits"
        width="140"
        align="center"
      />
      <el-table-column
        label="实收积分"
        prop="credits"
        width="110"
        align="center"
      />
      <el-table-column
        label="原因"
        prop="reason"
        min-width="180"
        show-overflow-tooltip
      />
      <el-table-column label="状态" width="100" align="center">
        <template #default="{ row }">
          <el-tag :type="statusTag(row.status).type" size="small">
            {{ statusTag(row.status).label }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="时间" prop="createTime" width="160" />
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

const anomalyTypeMap: Record<string, string> = {
  anomalous: "异常计费",
  manual: "人工调整",
  auto_compensated: "自动补偿",
};

function anomalyTypeLabel(type: string) {
  return anomalyTypeMap[type] ?? type;
}

function statusTag(status?: string) {
  switch (status) {
    case "compensated":
      return { label: "已补偿", type: "success" as const };
    case "ignored":
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
