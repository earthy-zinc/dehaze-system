<!-- 资源消耗：按模型/智能体/用户聚合（与计费口径一致）+ 日趋势 -->
<template>
  <el-card v-loading="store.costLoading" shadow="never">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>资源消耗</span>
        <div class="flex items-center gap-2">
          <el-radio-group
            v-model="store.costDimension"
            size="small"
            @change="handleDimensionChange"
          >
            <el-radio-button value="model">按模型</el-radio-button>
            <el-radio-button value="agent">按智能体</el-radio-button>
            <el-radio-button value="user">按用户</el-radio-button>
          </el-radio-group>
          <el-radio-group
            v-model="store.costRange"
            size="small"
            @change="handleRangeChange"
          >
            <el-radio-button :value="7">近7天</el-radio-button>
            <el-radio-button :value="30">近30天</el-radio-button>
            <el-radio-button :value="90">近90天</el-radio-button>
          </el-radio-group>
        </div>
      </div>
    </template>

    <el-empty
      v-if="
        !store.costItems.length && !store.costTrend.length && !store.costLoading
      "
      description="暂无消耗数据"
      :image-size="90"
    />
    <template v-else>
      <el-table :data="store.costItems" size="small">
        <el-table-column :label="dimensionLabel" min-width="160">
          <template #default="{ row }">
            {{ dimensionText(row as AiObservabilityCostItem) }}
          </template>
        </el-table-column>
        <el-table-column
          label="调用数"
          prop="traceCount"
          width="90"
          align="center"
        />
        <el-table-column label="总Token" width="110" align="center">
          <template #default="{ row }">{{
            fmtTokens(row.totalTokens)
          }}</template>
        </el-table-column>
        <el-table-column label="输入" width="100" align="center">
          <template #default="{ row }">{{
            fmtTokens(row.promptTokens)
          }}</template>
        </el-table-column>
        <el-table-column label="输出" width="100" align="center">
          <template #default="{ row }">{{
            fmtTokens(row.completionTokens)
          }}</template>
        </el-table-column>
        <el-table-column label="缓存命中" width="100" align="center">
          <template #default="{ row }">{{
            fmtTokens(row.cachedTokens)
          }}</template>
        </el-table-column>
      </el-table>

      <pagination
        v-if="store.costTotal > store.costPageSize"
        v-model:limit="store.costPageSize"
        v-model:page="store.costPageNum"
        v-model:total="store.costTotal"
        layout="prev, pager, next"
        @pagination="store.fetchCosts()"
      />

      <div ref="trendChartEl" class="chart" />
    </template>
  </el-card>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import type { AiObservabilityCostItem } from "dehaze-sdk-js";
import { fmtTokens } from "../format";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "CostPanel" });

const store = useAdminObservabilityStore();

const dimensionLabel = computed(
  () =>
    ({ model: "模型", agent: "智能体", user: "用户ID" })[store.costDimension]
);

function dimensionText(row: AiObservabilityCostItem) {
  if (store.costDimension === "model") return row.model ?? "-";
  if (store.costDimension === "agent") return row.agentCode ?? "-";
  return String(row.userId ?? "-");
}

function handleDimensionChange() {
  store.costPageNum = 1;
  store.fetchCosts();
}

function handleRangeChange() {
  store.costPageNum = 1;
  store.fetchCosts();
}

const trendChartEl = ref<HTMLDivElement>();
let trendChart: echarts.ECharts | null = null;

function renderTrend() {
  if (!trendChart) return;
  const trend = store.costTrend;
  trendChart.setOption(
    {
      title: {
        text: "日 Token 消耗",
        left: 10,
        top: 5,
        textStyle: { fontSize: 13, fontWeight: 500 },
      },
      tooltip: { trigger: "axis" },
      legend: { bottom: 0 },
      grid: {
        left: "2%",
        right: "3%",
        top: 40,
        bottom: 36,
        containLabel: true,
      },
      xAxis: { type: "category", data: trend.map((item) => item.date) },
      yAxis: [
        { type: "value", name: "Token" },
        { type: "value", name: "调用数" },
      ],
      series: [
        {
          name: "总Token",
          type: "bar",
          data: trend.map((item) => item.totalTokens),
          itemStyle: { color: "#409eff" },
        },
        {
          name: "调用数",
          type: "line",
          yAxisIndex: 1,
          smooth: true,
          data: trend.map((item) => item.traceCount),
          itemStyle: { color: "#67c23a" },
        },
      ],
    },
    true
  );
}

watch(
  () => store.costTrend,
  async (trend) => {
    if (!trend.length) return;
    await nextTick();
    trendChart ??= echarts.init(trendChartEl.value as HTMLDivElement);
    renderTrend();
  },
  { immediate: true, deep: true }
);

function resizeChart() {
  trendChart?.resize();
}

onMounted(() => {
  window.addEventListener("resize", resizeChart);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", resizeChart);
  trendChart?.dispose();
  trendChart = null;
});
</script>

<style lang="scss" scoped>
.chart {
  width: 100%;
  height: 240px;
  margin-top: 8px;
}
</style>
