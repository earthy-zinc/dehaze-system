<!-- 消耗与成本下钻：维度切换 + Top N 排行 + 用户级计费明细/流水 -->
<template>
  <el-card shadow="never">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>消耗与成本下钻</span>
        <el-radio-group
          v-model="billingStore.drilldownDimension"
          @change="billingStore.fetchDrilldown()"
        >
          <el-radio-button value="user">用户</el-radio-button>
          <el-radio-button value="model">模型</el-radio-button>
          <el-radio-button value="provider">供应商</el-radio-button>
          <el-radio-button value="day">时间</el-radio-button>
        </el-radio-group>
      </div>
    </template>

    <!-- 用户/模型/时间维度：积分消耗统计 -->
    <el-table
      v-if="billingStore.drilldownDimension !== 'provider'"
      v-loading="billingStore.drilldownLoading"
      :data="billingStore.drilldownData"
      size="small"
      highlight-current-row
      @row-click="handleRowClick"
    >
      <el-table-column
        :label="dimensionLabel"
        prop="dimension"
        min-width="140"
      />
      <el-table-column
        label="积分消耗"
        prop="totalCredits"
        sortable
        width="120"
        align="center"
      />
      <el-table-column
        label="输入Token"
        prop="totalInputTokens"
        sortable
        width="120"
        align="center"
      />
      <el-table-column
        label="输出Token"
        prop="totalOutputTokens"
        sortable
        width="120"
        align="center"
      />
      <el-table-column label="缓存命中率" width="110" align="center">
        <template #default="{ row }"
          >{{ (row.cacheHitRate * 100).toFixed(1) }}%</template
        >
      </el-table-column>
      <el-table-column
        label="缓存节省"
        prop="creditsSaved"
        sortable
        width="110"
        align="center"
      />
      <el-table-column
        label="降级次数"
        prop="degradationCount"
        sortable
        width="110"
        align="center"
      />
      <el-table-column
        v-if="billingStore.drilldownDimension === 'user'"
        label="操作"
        width="100"
        align="center"
      >
        <template #default="{ row }">
          <el-button
            link
            type="primary"
            size="small"
            @click.stop="selectUser(row.dimension)"
          >
            明细
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 供应商维度：成本-利润统计 -->
    <el-table
      v-else
      v-loading="billingStore.drilldownLoading"
      :data="billingStore.providerStats"
      size="small"
    >
      <el-table-column label="统计项" prop="dimension" min-width="120" />
      <el-table-column label="口径" width="100" align="center">
        <template #default="{ row }">
          <el-tag
            :type="row.metric === 'overall' ? 'primary' : 'info'"
            size="small"
          >
            {{ row.metric === "overall" ? "整体" : "AI参考" }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column
        label="收入（元）"
        prop="revenue"
        sortable
        width="120"
        align="center"
      />
      <el-table-column
        label="成本（元）"
        prop="cost"
        sortable
        width="120"
        align="center"
      />
      <el-table-column
        label="毛利（元）"
        prop="profit"
        sortable
        width="120"
        align="center"
      >
        <template #default="{ row }">
          <span :class="row.profit < 0 ? 'text-red-500' : ''">{{
            row.profit.toFixed(2)
          }}</span>
        </template>
      </el-table-column>
      <el-table-column label="毛利率" width="100" align="center">
        <template #default="{ row }"
          >{{ (row.profitRate * 100).toFixed(2) }}%</template
        >
      </el-table-column>
    </el-table>

    <!-- 时间维度：每日积分消耗趋势 -->
    <div
      v-show="billingStore.drilldownDimension === 'day'"
      ref="trendRef"
      class="h-[260px] w-full mt-2"
    />

    <!-- 用户级计费明细下钻 -->
    <template v-if="billingStore.selectedUserId != null">
      <el-divider content-position="left">
        用户 {{ billingStore.selectedUserId }} 计费明细
      </el-divider>
      <div class="mb-[12px] max-w-[420px]">
        <balance-quota-card :scope="billingStore.selectedUserId" />
      </div>
      <el-tabs model-value="records">
        <el-tab-pane label="计费明细" name="records">
          <billing-record-table :scope="billingStore.selectedUserId" />
        </el-tab-pane>
        <el-tab-pane label="积分流水" name="logs" lazy>
          <credit-log-table :scope="billingStore.selectedUserId" />
        </el-tab-pane>
      </el-tabs>
    </template>
  </el-card>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import { useAdminBillingStore } from "@/store/modules/adminBilling";
import BalanceQuotaCard from "@/components/billing/BalanceQuotaCard.vue";
import BillingRecordTable from "@/components/billing/BillingRecordTable.vue";
import CreditLogTable from "@/components/billing/CreditLogTable.vue";

defineOptions({ name: "AdminDrilldownPanel" });

const billingStore = useAdminBillingStore();

const dimensionLabel = computed(() => {
  switch (billingStore.drilldownDimension) {
    case "user":
      return "用户ID";
    case "model":
      return "模型";
    case "day":
      return "日期";
    default:
      return "维度";
  }
});

const trendRef = ref<HTMLDivElement>();
let trendChart: echarts.ECharts | null = null;

function renderTrend() {
  if (billingStore.drilldownDimension !== "day" || !trendRef.value) return;
  if (!trendChart) {
    trendChart = echarts.init(trendRef.value);
  }
  const data = billingStore.drilldownData;
  trendChart.setOption(
    {
      tooltip: { trigger: "axis" },
      grid: { left: "3%", right: "3%", bottom: "10%", containLabel: true },
      xAxis: { type: "category", data: data.map((row) => row.dimension) },
      yAxis: { type: "value", name: "积分" },
      series: [
        {
          name: "每日积分消耗",
          type: "line",
          smooth: true,
          data: data.map((row) => row.totalCredits),
        },
      ],
    },
    true
  );
}

watch(
  () => [billingStore.drilldownDimension, billingStore.drilldownData],
  () => nextTick(renderTrend),
  { deep: true }
);

function selectUser(dimension: string) {
  billingStore.selectedUserId = Number(dimension);
}

function handleRowClick(row: { dimension: string }) {
  if (billingStore.drilldownDimension === "user") {
    selectUser(row.dimension);
  }
}

onBeforeUnmount(() => {
  trendChart?.dispose();
  trendChart = null;
});

onMounted(() => {
  billingStore.fetchDrilldown();
});
</script>
