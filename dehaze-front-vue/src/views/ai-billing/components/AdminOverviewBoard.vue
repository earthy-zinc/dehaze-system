<!-- 概览看板：周期切换 + 双口径指标卡 + 毛利趋势 + 负毛利预警 -->
<template>
  <el-card v-loading="billingStore.overviewLoading" shadow="never">
    <template #header>
      <div class="flex justify-between items-center flex-wrap gap-2">
        <span>经营概览</span>
        <div class="flex items-center gap-2">
          <el-radio-group
            v-model="billingStore.overviewPeriod"
            @change="handlePeriodChange"
          >
            <el-radio-button value="month">本月</el-radio-button>
            <el-radio-button value="quarter">本季</el-radio-button>
            <el-radio-button value="custom">自定义</el-radio-button>
          </el-radio-group>
          <el-date-picker
            v-if="billingStore.overviewPeriod === 'custom'"
            v-model="customRange"
            type="daterange"
            value-format="YYYY-MM-DD"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            @change="handleCustomRange"
          />
          <el-button
            :loading="billingStore.overviewLoading"
            @click="billingStore.fetchOverview()"
          >
            <el-icon><Refresh /></el-icon>刷新
          </el-button>
        </div>
      </div>
    </template>

    <!-- 负毛利预警：毛利 < 0 时顶部展示，点击进入下钻明细 -->
    <el-alert
      v-if="grossProfit != null && grossProfit < 0"
      class="mb-[12px]"
      type="error"
      :closable="false"
      :title="`当前周期毛利为负（${grossProfit.toFixed(2)} 元），点击查看消耗明细`"
      @click="emit('drillDetail')"
    />

    <!-- 整体口径（官方）主指标卡 -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-[12px] mb-[12px]">
      <el-card v-if="overall" shadow="never">
        <div class="text-sm text-gray-500">实收收入（元）</div>
        <div class="text-2xl font-semibold mt-1">
          {{ overall.revenue.toFixed(2) }}
        </div>
      </el-card>
      <el-card v-if="overall" shadow="never">
        <div class="text-sm text-gray-500">模型调用成本（元）</div>
        <div class="text-2xl font-semibold mt-1">
          {{ overall.cost.toFixed(2) }}
        </div>
      </el-card>
      <el-card v-if="overall" shadow="never">
        <div class="text-sm text-gray-500">毛利（元）</div>
        <div
          class="text-2xl font-semibold mt-1"
          :class="overall.profit < 0 ? 'text-red-500' : ''"
        >
          {{ overall.profit.toFixed(2) }}
        </div>
      </el-card>
      <el-card v-if="overall" shadow="never">
        <div class="text-sm text-gray-500">毛利率</div>
        <div
          class="text-2xl font-semibold mt-1"
          :class="overall.profitRate < 0 ? 'text-red-500' : ''"
        >
          {{ (overall.profitRate * 100).toFixed(2) }}%
        </div>
      </el-card>
    </div>
    <el-empty
      v-if="!overall && !billingStore.overviewLoading"
      description="暂无成本-利润统计数据"
      :image-size="60"
    />

    <!-- AI 参考口径（辅助）折叠展示 -->
    <el-collapse v-if="aiStatRow" class="mb-[12px]">
      <el-collapse-item title="AI 参考口径（辅助定价决策）">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-[12px]">
          <div>
            <div class="text-xs text-gray-400">AI 分摊收入（元）</div>
            <div class="text-lg mt-1">{{ aiStatRow.revenue.toFixed(2) }}</div>
          </div>
          <div>
            <div class="text-xs text-gray-400">模型调用成本（元）</div>
            <div class="text-lg mt-1">{{ aiStatRow.cost.toFixed(2) }}</div>
          </div>
          <div>
            <div class="text-xs text-gray-400">AI 参考毛利（元）</div>
            <div
              class="text-lg mt-1"
              :class="aiStatRow.profit < 0 ? 'text-red-500' : ''"
            >
              {{ aiStatRow.profit.toFixed(2) }}
            </div>
          </div>
          <div>
            <div class="text-xs text-gray-400">AI 参考毛利率</div>
            <div class="text-lg mt-1">
              {{ (aiStatRow.profitRate * 100).toFixed(2) }}%
            </div>
          </div>
        </div>
      </el-collapse-item>
    </el-collapse>

    <div ref="trendRef" class="h-[280px] w-full" />
  </el-card>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import { Refresh } from "@element-plus/icons-vue";
import { useAdminBillingStore } from "@/store/modules/adminBilling";

defineOptions({ name: "AdminOverviewBoard" });

const emit = defineEmits<{ (e: "drillDetail"): void }>();

const billingStore = useAdminBillingStore();

const overall = computed(() => billingStore.overallStat);
const aiStatRow = computed(() => billingStore.aiStat);
const grossProfit = computed(() => overall.value?.profit ?? null);

const customRange = ref<[string, string]>(["", ""]);

function handlePeriodChange() {
  if (billingStore.overviewPeriod !== "custom") {
    billingStore.fetchOverview();
  }
}

function handleCustomRange() {
  if (customRange.value && customRange.value[0]) {
    billingStore.periodRange = customRange.value;
    billingStore.fetchOverview();
  }
}

// 毛利与成本趋势图：按成本-利润统计双口径渲染
const trendRef = ref<HTMLDivElement>();
let trendChart: echarts.ECharts | null = null;

function renderTrend() {
  if (!trendRef.value) return;
  if (!trendChart) {
    trendChart = echarts.init(trendRef.value);
  }
  const rows = billingStore.overview;
  const categories = [...new Set(rows.map((row) => row.dimension || "汇总"))];
  const metricLabel = { overall: "整体", ai: "AI参考" } as const;
  const series = (["overall", "ai"] as const).flatMap((metric) => {
    const metricRows = rows.filter((row) => row.metric === metric);
    if (metricRows.length === 0) return [];
    const byDimension = new Map(
      metricRows.map((row) => [row.dimension || "汇总", row])
    );
    return [
      {
        name: `${metricLabel[metric]}成本`,
        type: "bar" as const,
        data: categories.map((d) => byDimension.get(d)?.cost ?? 0),
      },
      {
        name: `${metricLabel[metric]}毛利`,
        type: "line" as const,
        data: categories.map((d) => byDimension.get(d)?.profit ?? 0),
      },
    ];
  });

  trendChart.setOption(
    {
      tooltip: { trigger: "axis" },
      legend: { bottom: 0 },
      grid: { left: "3%", right: "3%", bottom: "15%", containLabel: true },
      xAxis: { type: "category", data: categories },
      yAxis: { type: "value" },
      series,
    },
    true
  );
}

watch(
  () => billingStore.overview,
  () => renderTrend(),
  { deep: true }
);

onBeforeUnmount(() => {
  trendChart?.dispose();
  trendChart = null;
});

onMounted(() => {
  billingStore.fetchOverview();
});
</script>
