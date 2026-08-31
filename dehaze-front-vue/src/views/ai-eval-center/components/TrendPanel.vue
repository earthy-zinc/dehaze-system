<!-- 评测历史趋势：各智能体总分随评测时间变化 -->
<template>
  <el-card shadow="never" class="mb-[12px]">
    <template #header>
      <div class="flex justify-between items-center">
        <span>评测历史趋势</span>
        <span class="text-xs text-gray-400">
          已完成评测 {{ evalStore.trendData.length }} 条
        </span>
      </div>
    </template>

    <el-empty
      v-if="evalStore.trendData.length === 0 && !evalStore.trendLoading"
      description="所选范围内暂无评测趋势数据"
      :image-size="60"
    />
    <div
      v-show="evalStore.trendData.length > 0"
      ref="chartRef"
      class="h-[280px] w-full"
    />
  </el-card>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import { useAdminEvalStore } from "@/store/modules/adminEval";
import { formatTime } from "../eval-meta";

defineOptions({ name: "TrendPanel" });

const evalStore = useAdminEvalStore();

const chartRef = ref<HTMLDivElement>();
let chart: echarts.ECharts | null = null;

const categories = computed(() => {
  const times = evalStore.trendData.map((item) => formatTime(item.createTime));
  return [...new Set(times)].sort();
});

const series = computed(() => {
  const byAgent = new Map<string, Map<string, number>>();
  evalStore.trendData.forEach((item) => {
    const agent = item.agentName ?? "-";
    if (!byAgent.has(agent)) {
      byAgent.set(agent, new Map());
    }
    if (item.totalScore != null) {
      byAgent.get(agent)!.set(formatTime(item.createTime), item.totalScore);
    }
  });
  return [...byAgent.entries()].map(([name, points]) => ({
    name,
    type: "line" as const,
    smooth: true,
    connectNulls: true,
    data: categories.value.map((time) => points.get(time) ?? null),
  }));
});

function renderChart() {
  if (evalStore.trendData.length === 0 || !chartRef.value) return;
  if (!chart) {
    chart = echarts.init(chartRef.value);
  }
  chart.setOption(
    {
      tooltip: { trigger: "axis" },
      legend: { bottom: 0 },
      grid: { left: "3%", right: "3%", bottom: "15%", containLabel: true },
      xAxis: {
        type: "category",
        data: categories.value,
        axisLabel: { rotate: 30 },
      },
      yAxis: { type: "value", name: "总分", min: 0, max: 100 },
      series: series.value,
    },
    true
  );
}

watch(
  () => evalStore.trendData,
  () => nextTick(renderChart),
  { deep: true }
);

onBeforeUnmount(() => {
  chart?.dispose();
  chart = null;
});
</script>
