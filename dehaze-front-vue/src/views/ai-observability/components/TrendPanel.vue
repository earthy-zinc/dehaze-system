<!-- 性能趋势：成功率按模型/智能体分线；延迟为全量按调用量加权均值 -->
<template>
  <el-card v-loading="store.trendLoading" shadow="never">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>性能趋势</span>
        <div class="flex items-center gap-2">
          <el-radio-group
            v-model="store.trendDimension"
            size="small"
            @change="store.fetchTrends()"
          >
            <el-radio-button value="model">按模型</el-radio-button>
            <el-radio-button value="agent">按智能体</el-radio-button>
          </el-radio-group>
          <el-radio-group
            v-model="store.trendRange"
            size="small"
            @change="store.fetchTrends()"
          >
            <el-radio-button :value="7">近7天</el-radio-button>
            <el-radio-button :value="30">近30天</el-radio-button>
            <el-radio-button :value="90">近90天</el-radio-button>
          </el-radio-group>
        </div>
      </div>
    </template>

    <el-empty
      v-if="!store.trendData.length"
      description="暂无趋势数据"
      :image-size="90"
    />
    <template v-else>
      <div ref="successChartEl" class="chart" />
      <div ref="latencyChartEl" class="chart" />
    </template>
  </el-card>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import type { AiObservabilityTrendItem } from "dehaze-sdk-js";
import { useAdminObservabilityStore } from "@/store/modules/adminObservability";

defineOptions({ name: "TrendPanel" });

const store = useAdminObservabilityStore();

const successChartEl = ref<HTMLDivElement>();
const latencyChartEl = ref<HTMLDivElement>();
let successChart: echarts.ECharts | null = null;
let latencyChart: echarts.ECharts | null = null;

function seriesName(row: AiObservabilityTrendItem) {
  return store.trendDimension === "model"
    ? (row.model ?? "未知模型")
    : (row.agentCode ?? "默认智能体");
}

function renderCharts() {
  const rows = store.trendData;
  if (!successChart || !latencyChart) return;

  const dates = [...new Set(rows.map((row) => row.date))].sort();
  const names = [...new Set(rows.map(seriesName))];

  const successSeries = names.map((name) => ({
    name,
    type: "line" as const,
    smooth: true,
    showSymbol: dates.length <= 15,
    data: dates.map((date) => {
      const row = rows.find((r) => seriesName(r) === name && r.date === date);
      return row ? row.successRate : null;
    }),
  }));

  successChart.setOption(
    {
      title: {
        text: "成功率（%）",
        left: 10,
        top: 5,
        textStyle: { fontSize: 13, fontWeight: 500 },
      },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value: number) => `${value}%`,
      },
      legend: { bottom: 0, type: "scroll" },
      grid: {
        left: "2%",
        right: "3%",
        top: 40,
        bottom: 36,
        containLabel: true,
      },
      xAxis: { type: "category", boundaryGap: false, data: dates },
      yAxis: { type: "value", max: 100 },
      series: successSeries,
    },
    true
  );

  const latencySeries = [
    {
      name: "平均首Token延迟",
      pick: (row: AiObservabilityTrendItem) => row.avgFirstTokenMs,
    },
    {
      name: "平均总耗时",
      pick: (row: AiObservabilityTrendItem) => row.avgDurationMs,
    },
  ].map(({ name, pick }) => ({
    name,
    type: "line" as const,
    smooth: true,
    showSymbol: dates.length <= 15,
    data: dates.map((date) => {
      // 延迟全量聚合口径：按各维度调用量加权平均
      const dayRows = rows.filter((row) => row.date === date);
      const calls = dayRows.reduce((sum, row) => sum + row.callCount, 0);
      if (calls === 0) return null;
      return (
        dayRows.reduce(
          (sum, row) => sum + row.callCount * (pick(row) ?? 0),
          0
        ) / calls
      );
    }),
  }));

  latencyChart.setOption(
    {
      title: {
        text: "平均延迟（ms，全量加权）",
        left: 10,
        top: 5,
        textStyle: { fontSize: 13, fontWeight: 500 },
      },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value: number) => `${Math.round(value)}ms`,
      },
      legend: { bottom: 0, type: "scroll" },
      grid: {
        left: "2%",
        right: "3%",
        top: 40,
        bottom: 36,
        containLabel: true,
      },
      xAxis: { type: "category", boundaryGap: false, data: dates },
      yAxis: { type: "value" },
      series: latencySeries,
    },
    true
  );
}

watch(
  () => store.trendData,
  async (data) => {
    if (!data.length) return;
    await nextTick();
    successChart ??= echarts.init(successChartEl.value as HTMLDivElement);
    latencyChart ??= echarts.init(latencyChartEl.value as HTMLDivElement);
    renderCharts();
  },
  { immediate: true, deep: true }
);

function resizeCharts() {
  successChart?.resize();
  latencyChart?.resize();
}

onMounted(() => {
  window.addEventListener("resize", resizeCharts);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", resizeCharts);
  successChart?.dispose();
  latencyChart?.dispose();
  successChart = null;
  latencyChart = null;
});
</script>

<style lang="scss" scoped>
.chart {
  width: 100%;
  height: 260px;

  & + & {
    margin-top: 8px;
  }
}
</style>
