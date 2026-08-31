<!-- 消耗总览区：日/月消耗趋势、模型消耗分布、缓存节省汇总 -->
<script lang="ts" setup>
import { BillingModelDistVO, BillingSummaryVO } from "dehaze-sdk-js";
import { Coin } from "@element-plus/icons-vue";
import * as echarts from "echarts";
import {
  computed,
  nextTick,
  onActivated,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { SummaryDimension, useBillingStore } from "@/store/modules/billing";

defineOptions({ name: "ConsumptionOverview" });

// 模型超过该数量后其余归入"其他"，避免环形图被长尾模型切碎
const TOP_MODEL_COUNT = 5;

const billingStore = useBillingStore();

const trendChartEl = ref<HTMLDivElement>();
const distChartEl = ref<HTMLDivElement>();
let trendChart: echarts.ECharts | null = null;
let distChart: echarts.ECharts | null = null;

// el-radio-group 的回调值类型是 string | number | boolean，维度取值需在此收敛
const dimension = computed({
  get: () => billingStore.summaryDimension,
  set: (value: string | number | boolean | undefined) =>
    billingStore.setDimension(value as SummaryDimension),
});

const trend = computed(() => billingStore.consumptionSummary?.trend ?? []);
const savings = computed(
  () => billingStore.consumptionSummary?.savings?.creditsSaved ?? 0
);
const distribution = computed<BillingModelDistVO[]>(() => {
  const list = [...(billingStore.consumptionSummary?.modelDistribution ?? [])]
    .filter((item) => item.credits > 0)
    .sort((a, b) => b.credits - a.credits);
  if (list.length <= TOP_MODEL_COUNT) return list;
  const rest = list.slice(TOP_MODEL_COUNT);
  return [
    ...list.slice(0, TOP_MODEL_COUNT),
    {
      model: "其他",
      credits: rest.reduce((sum, item) => sum + item.credits, 0),
      tokens: rest.reduce((sum, item) => sum + item.tokens, 0),
    },
  ];
});

const hasData = computed(
  () => trend.value.length > 0 || distribution.value.length > 0
);

function renderTrendChart(summary: BillingSummaryVO | null) {
  if (!trendChart) return;
  const points = summary?.trend ?? [];
  trendChart.setOption({
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: number) => `${value} 积分`,
    },
    grid: {
      left: "2%",
      right: "3%",
      top: "12%",
      bottom: "2%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: points.map((point) => point.date),
      axisLine: { lineStyle: { color: "#dcdfe6" } },
      axisLabel: { color: "#909399" },
    },
    yAxis: {
      type: "value",
      name: "积分",
      nameTextStyle: { color: "#909399" },
      axisLabel: { color: "#909399" },
      splitLine: { lineStyle: { type: "dashed", color: "#ebeef5" } },
    },
    series: [
      {
        name: "消耗积分",
        type: "line",
        smooth: true,
        showSymbol: points.length <= 15,
        areaStyle: { opacity: 0.15 },
        itemStyle: { color: "#409eff" },
        data: points.map((point) => point.credits),
      },
    ],
  });
}

function renderDistChart() {
  if (!distChart) return;
  distChart.setOption({
    tooltip: { trigger: "item", formatter: "{b}：{c} 积分（{d}%）" },
    legend: { bottom: 0, textStyle: { color: "#909399" } },
    series: [
      {
        type: "pie",
        radius: ["45%", "68%"],
        center: ["50%", "45%"],
        itemStyle: { borderColor: "#fff", borderWidth: 2 },
        label: { formatter: "{b}\n{d}%", color: "#606266" },
        data: distribution.value.map((item) => ({
          name: item.model,
          value: item.credits,
        })),
      },
    ],
  });
}

function resizeCharts() {
  trendChart?.resize();
  distChart?.resize();
}

// 图表容器随数据就绪才渲染，故在首帧数据到后再初始化
watch(
  () => billingStore.consumptionSummary,
  async (summary) => {
    if (!hasData.value) return;
    await nextTick();
    trendChart ??= echarts.init(trendChartEl.value as HTMLDivElement);
    distChart ??= echarts.init(distChartEl.value as HTMLDivElement);
    renderTrendChart(summary);
    renderDistChart();
  },
  { immediate: true }
);

onMounted(() => {
  window.addEventListener("resize", resizeCharts);
});

onActivated(resizeCharts);

onBeforeUnmount(() => {
  window.removeEventListener("resize", resizeCharts);
  trendChart?.dispose();
  distChart?.dispose();
  trendChart = null;
  distChart = null;
});
</script>

<template>
  <el-card
    v-loading="billingStore.loading"
    shadow="never"
    class="overview-card"
  >
    <template #header>
      <div class="card-header">
        <span class="card-title">消耗总览</span>
        <el-radio-group v-model="dimension" size="small">
          <el-radio-button value="day">按日</el-radio-button>
          <el-radio-button value="month">按月</el-radio-button>
        </el-radio-group>
      </div>
    </template>

    <el-empty v-if="!hasData" description="暂无消耗数据" :image-size="90" />
    <template v-else>
      <div class="chart-row">
        <div ref="trendChartEl" class="chart trend-chart" />
        <div ref="distChartEl" class="chart dist-chart" />
      </div>
      <div class="savings-row">
        <el-icon class="savings-icon"><Coin /></el-icon>
        <span>
          缓存命中累计为你节省
          <strong class="savings-value">{{ savings }}</strong>
          积分
        </span>
      </div>
    </template>
  </el-card>
</template>

<style lang="scss" scoped>
.overview-card {
  border-radius: 12px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .chart-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .chart {
    height: 280px;
  }

  .trend-chart {
    flex: 1 1 420px;
    min-width: 300px;
  }

  .dist-chart {
    flex: 1 1 300px;
    min-width: 260px;
  }

  .savings-row {
    display: flex;
    gap: 8px;
    align-items: center;
    padding-top: 14px;
    margin-top: 8px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
    border-top: 1px dashed var(--el-border-color);

    .savings-icon {
      color: var(--el-color-warning);
    }

    .savings-value {
      margin: 0 2px;
      font-size: 16px;
      color: var(--el-color-warning);
    }
  }
}
</style>
