<template>
  <div class="app-container">
    <div class="search-container">
      <el-form :inline="true">
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="timeRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
            value-format="YYYY-MM-DD"
            :shortcuts="shortcuts"
            style="width: 260px"
            @change="handleTimeChange"
          />
        </el-form-item>
        <el-form-item>
          <el-radio-group v-model="activeTab" @change="handleTabChange">
            <el-radio-button value="rating">
              <el-icon><Star /></el-icon><span class="tab-text">评价统计</span>
            </el-radio-button>
            <el-radio-button value="feedback">
              <el-icon><ChatLineRound /></el-icon
              ><span class="tab-text">反馈统计</span>
            </el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>
    </div>

    <div v-if="activeTab === 'rating'" v-loading="ratingLoading">
      <el-row :gutter="16">
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">总评价数</div>
              <div class="stat-value">{{ totalRatings }}</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">平均评分</div>
              <div class="stat-value">{{ averageRating.toFixed(2) }}</div>
              <el-rate :model-value="averageRating" disabled allow-half />
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">好评率</div>
              <div class="stat-value" style="color: #52c41a">
                {{ positiveRate }}%
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">差评率</div>
              <div class="stat-value" style="color: #f5222d">
                {{ negativeRate }}%
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" class="section-card">
        <template #header>
          <div class="card-title">
            <el-icon><DataAnalysis /></el-icon>评分分布
          </div>
        </template>
        <div id="ratingDistChart" style="width: 100%; height: 300px"></div>
      </el-card>

      <el-card shadow="never" class="section-card">
        <template #header>
          <div class="card-title">
            <el-icon><DataAnalysis /></el-icon>算法维度统计
          </div>
        </template>
        <el-table :data="ratingStats?.algorithmStats || []" border>
          <el-table-column label="算法" prop="algorithmName" min-width="160" />
          <el-table-column label="平均评分" width="180" align="center">
            <template #default="scope">
              <el-rate
                :model-value="scope.row.averageRating"
                disabled
                allow-half
              />
            </template>
          </el-table-column>
          <el-table-column
            label="评价数"
            prop="totalRatings"
            width="100"
            align="center"
          />
          <el-table-column label="差评率" min-width="220">
            <template #default="scope">
              <el-progress
                :percentage="scope.row.lowRatingRate"
                :color="scope.row.lowRatingRate > 20 ? '#f5222d' : '#409eff'"
              />
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-row :gutter="16" class="section-card">
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="card-title">正面标签排行</div>
            </template>
            <el-table :data="ratingStats?.positiveTagRanking || []" border>
              <el-table-column label="标签" prop="tag" min-width="120" />
              <el-table-column
                label="次数"
                prop="count"
                width="80"
                align="center"
              />
              <el-table-column label="相对占比" min-width="180">
                <template #default="scope">
                  <el-progress
                    :percentage="
                      Math.round((scope.row.count * 100) / positiveMax)
                    "
                    color="#52c41a"
                  />
                </template>
              </el-table-column>
            </el-table>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="card-title">负面标签排行</div>
            </template>
            <el-table :data="ratingStats?.negativeTagRanking || []" border>
              <el-table-column label="标签" prop="tag" min-width="120" />
              <el-table-column
                label="次数"
                prop="count"
                width="80"
                align="center"
              />
              <el-table-column label="相对占比" min-width="180">
                <template #default="scope">
                  <el-progress
                    :percentage="
                      Math.round((scope.row.count * 100) / negativeMax)
                    "
                    color="#f5222d"
                  />
                </template>
              </el-table-column>
            </el-table>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <div v-if="activeTab === 'feedback'" v-loading="feedbackLoading">
      <el-row :gutter="16">
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">总反馈数</div>
              <div class="stat-value">{{ totalFeedback }}</div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">平均响应时间</div>
              <div class="stat-value">
                {{ feedbackStats?.averageResponseTime ?? 0
                }}<span class="unit">分钟</span>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">平均关闭时间</div>
              <div class="stat-value">
                {{ feedbackStats?.averageCloseTime ?? 0
                }}<span class="unit">小时</span>
              </div>
            </div>
          </el-card>
        </el-col>
        <el-col :span="6">
          <el-card shadow="hover">
            <div class="stat-card">
              <div class="stat-label">待处理数</div>
              <div class="stat-value" style="color: #e6a23c">
                {{ pendingCount }}
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="16" class="section-card">
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="card-title">
                <el-icon><DataAnalysis /></el-icon>类型分布
              </div>
            </template>
            <div
              id="feedbackTypeChart"
              style="width: 100%; height: 300px"
            ></div>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              <div class="card-title">
                <el-icon><DataAnalysis /></el-icon>状态分布
              </div>
            </template>
            <div
              id="feedbackStatusChart"
              style="width: 100%; height: 300px"
            ></div>
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" class="section-card">
        <template #header>
          <div class="card-title">
            <el-icon><DataAnalysis /></el-icon>模块分布
          </div>
        </template>
        <el-table :data="feedbackStats?.moduleDistribution || []" border>
          <el-table-column label="模块" prop="module" min-width="160" />
          <el-table-column
            label="反馈数"
            prop="count"
            width="100"
            align="center"
          />
          <el-table-column label="占比" min-width="240">
            <template #default="scope">
              <el-progress
                :percentage="
                  totalFeedback
                    ? Math.round((scope.row.count * 100) / totalFeedback)
                    : 0
                "
              />
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-card shadow="never" class="section-card">
        <template #header>
          <div class="card-title">
            <el-icon><DataAnalysis /></el-icon>高频关键词
          </div>
        </template>
        <div class="keyword-cloud">
          <el-tag
            v-for="kw in feedbackStats?.topKeywords || []"
            :key="kw.keyword"
            :style="{
              fontSize: keywordFontSize(kw.count) + 'px',
              margin: '6px',
            }"
            effect="plain"
          >
            {{ kw.keyword }} ({{ kw.count }})
          </el-tag>
          <span
            v-if="
              !feedbackStats?.topKeywords ||
              feedbackStats.topKeywords.length === 0
            "
            style="color: #909399"
            >暂无数据</span
          >
        </div>
      </el-card>
    </div>
  </div>
</template>

<script lang="ts" setup>
import * as echarts from "echarts";
import { Star, ChatLineRound, DataAnalysis } from "@element-plus/icons-vue";
import {
  FeedbackAPI,
  RatingStatsVO,
  FeedbackStatsVO,
  FeedbackType,
  FeedbackStatus,
} from "dehaze-sdk-js";

defineOptions({ name: "FeedbackStats" });

const route = useRoute();

const activeTab = ref<"rating" | "feedback">(
  route.query.tab === "feedback" ? "feedback" : "rating"
);

const ratingLoading = ref(false);
const feedbackLoading = ref(false);
const ratingStats = ref<RatingStatsVO | null>(null);
const feedbackStats = ref<FeedbackStatsVO | null>(null);

const ratingDistChart = ref<echarts.ECharts | null>(null);
const feedbackTypeChart = ref<echarts.ECharts | null>(null);
const feedbackStatusChart = ref<echarts.ECharts | null>(null);

function pad(n: number) {
  return String(n).padStart(2, "0");
}
function formatDate(d: Date) {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}
function defaultRange(): [string, string] {
  const end = new Date();
  const start = new Date();
  start.setTime(start.getTime() - 30 * 24 * 3600 * 1000);
  return [formatDate(start), formatDate(end)];
}
const timeRange = ref<[string, string]>(defaultRange());

const shortcuts = [
  {
    text: "近7天",
    value: () => {
      const end = new Date();
      const start = new Date();
      start.setTime(start.getTime() - 7 * 24 * 3600 * 1000);
      return [start, end];
    },
  },
  {
    text: "近30天",
    value: () => {
      const end = new Date();
      const start = new Date();
      start.setTime(start.getTime() - 30 * 24 * 3600 * 1000);
      return [start, end];
    },
  },
  {
    text: "近90天",
    value: () => {
      const end = new Date();
      const start = new Date();
      start.setTime(start.getTime() - 90 * 24 * 3600 * 1000);
      return [start, end];
    },
  },
];

const ratingDistribution = computed(
  () => ratingStats.value?.ratingDistribution || {}
);
const totalRatings = computed(() => ratingStats.value?.totalRatings || 0);
const averageRating = computed(() => ratingStats.value?.averageRating || 0);
const positiveRate = computed(() => {
  const d = ratingDistribution.value;
  const good = (d[4] || 0) + (d[5] || 0);
  return totalRatings.value
    ? Math.round((good * 10000) / totalRatings.value) / 100
    : 0;
});
const negativeRate = computed(() => {
  const d = ratingDistribution.value;
  const bad = (d[1] || 0) + (d[2] || 0);
  return totalRatings.value
    ? Math.round((bad * 10000) / totalRatings.value) / 100
    : 0;
});
const positiveMax = computed(() =>
  Math.max(
    1,
    ...(ratingStats.value?.positiveTagRanking || []).map((t) => t.count)
  )
);
const negativeMax = computed(() =>
  Math.max(
    1,
    ...(ratingStats.value?.negativeTagRanking || []).map((t) => t.count)
  )
);

const totalFeedback = computed(() => feedbackStats.value?.totalFeedback || 0);
const pendingCount = computed(
  () => feedbackStats.value?.statusDistribution?.pending || 0
);
const keywordMax = computed(() =>
  Math.max(1, ...(feedbackStats.value?.topKeywords || []).map((k) => k.count))
);
function keywordFontSize(count: number) {
  return Math.round(14 + (count / keywordMax.value) * 10);
}

const typeLabelMap: Record<FeedbackType, string> = {
  suggestion: "功能建议",
  bug: "问题报告",
  experience: "体验反馈",
  complaint: "投诉",
};
const typeColorMap: Record<FeedbackType, string> = {
  suggestion: "#409eff",
  experience: "#67c23a",
  complaint: "#e6a23c",
  bug: "#f56c6c",
};
const statusLabelMap: Record<FeedbackStatus, string> = {
  pending: "待处理",
  processing: "处理中",
  replied: "已回复",
  closed: "已关闭",
};
const statusColorMap: Record<FeedbackStatus, string> = {
  pending: "#e6a23c",
  processing: "#409eff",
  replied: "#67c23a",
  closed: "#909399",
};

function initRatingCharts(data: RatingStatsVO) {
  const distEl = document.getElementById("ratingDistChart");
  if (!distEl) return;
  if (ratingDistChart.value) {
    ratingDistChart.value.dispose();
    ratingDistChart.value = null;
  }
  ratingDistChart.value = markRaw(echarts.init(distEl));
  const colorMap: Record<number, string> = {
    1: "#f5222d",
    2: "#fa8c16",
    3: "#faad14",
    4: "#409eff",
    5: "#52c41a",
  };
  ratingDistChart.value.setOption({
    tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
    legend: { bottom: 0 },
    series: [
      {
        type: "pie",
        radius: ["40%", "70%"],
        label: { formatter: "{b}: {c}" },
        data: Object.entries(data.ratingDistribution).map(([k, v]) => ({
          name: k + "星",
          value: v,
          itemStyle: { color: colorMap[Number(k)] },
        })),
      },
    ],
  });
}

function initFeedbackCharts(data: FeedbackStatsVO) {
  const typeEl = document.getElementById("feedbackTypeChart");
  if (typeEl) {
    if (feedbackTypeChart.value) {
      feedbackTypeChart.value.dispose();
      feedbackTypeChart.value = null;
    }
    feedbackTypeChart.value = markRaw(echarts.init(typeEl));
    feedbackTypeChart.value.setOption({
      tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
      legend: { bottom: 0 },
      series: [
        {
          type: "pie",
          radius: ["40%", "70%"],
          label: { formatter: "{b}: {c}" },
          data: (Object.keys(typeLabelMap) as FeedbackType[]).map((t) => ({
            name: typeLabelMap[t],
            value: data.typeDistribution?.[t] || 0,
            itemStyle: { color: typeColorMap[t] },
          })),
        },
      ],
    });
  }

  const statusEl = document.getElementById("feedbackStatusChart");
  if (statusEl) {
    if (feedbackStatusChart.value) {
      feedbackStatusChart.value.dispose();
      feedbackStatusChart.value = null;
    }
    feedbackStatusChart.value = markRaw(echarts.init(statusEl));
    const statuses = Object.keys(statusLabelMap) as FeedbackStatus[];
    feedbackStatusChart.value.setOption({
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { bottom: 0 },
      grid: { left: "3%", right: "3%", bottom: "15%", containLabel: true },
      xAxis: {
        type: "category",
        data: statuses.map((s) => statusLabelMap[s]),
      },
      yAxis: { type: "value" },
      series: [
        {
          type: "bar",
          barWidth: "40%",
          data: statuses.map((s) => ({
            value: data.statusDistribution?.[s] || 0,
            itemStyle: { color: statusColorMap[s] },
          })),
        },
      ],
    });
  }
}

function disposeCharts() {
  [ratingDistChart, feedbackTypeChart, feedbackStatusChart].forEach((c) => {
    if (c.value) {
      c.value.dispose();
      c.value = null;
    }
  });
}

function handleResize() {
  [ratingDistChart, feedbackTypeChart, feedbackStatusChart].forEach((c) =>
    c.value?.resize()
  );
}

function loadRatingStats() {
  ratingLoading.value = true;
  FeedbackAPI.getRatingStats(timeRange.value[0], timeRange.value[1])
    .then((data) => {
      ratingStats.value = data;
      if (activeTab.value === "rating") {
        nextTick(() => initRatingCharts(data));
      }
    })
    .finally(() => {
      ratingLoading.value = false;
    });
}

function loadFeedbackStats() {
  feedbackLoading.value = true;
  FeedbackAPI.getFeedbackStats(timeRange.value[0], timeRange.value[1])
    .then((data) => {
      feedbackStats.value = data;
      if (activeTab.value === "feedback") {
        nextTick(() => initFeedbackCharts(data));
      }
    })
    .finally(() => {
      feedbackLoading.value = false;
    });
}

function handleTabChange() {
  disposeCharts();
  nextTick(() => {
    if (activeTab.value === "rating" && ratingStats.value) {
      initRatingCharts(ratingStats.value);
    } else if (activeTab.value === "feedback" && feedbackStats.value) {
      initFeedbackCharts(feedbackStats.value);
    }
  });
}

function handleTimeChange() {
  if (!timeRange.value || timeRange.value.length !== 2) {
    ElMessage.warning("请选择时间范围");
    return;
  }
  if (activeTab.value === "rating") {
    loadRatingStats();
  } else {
    loadFeedbackStats();
  }
}

onMounted(() => {
  loadRatingStats();
  loadFeedbackStats();
  window.addEventListener("resize", handleResize);
});

onActivated(() => {
  handleResize();
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", handleResize);
  disposeCharts();
});
</script>

<style lang="scss" scoped>
.search-container {
  margin-bottom: 16px;
}

.tab-text {
  margin-left: 4px;
}

.section-card {
  margin-top: 16px;
}

.stat-card {
  text-align: center;

  .stat-label {
    margin-bottom: 8px;
    font-size: 14px;
    color: #909399;
  }

  .stat-value {
    font-size: 28px;
    font-weight: 600;
    color: #303133;

    .unit {
      margin-left: 4px;
      font-size: 14px;
      font-weight: normal;
      color: #909399;
    }
  }
}

.card-title {
  display: flex;
  align-items: center;
  font-weight: 600;

  .el-icon {
    margin-right: 6px;
  }
}

.keyword-cloud {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  min-height: 60px;
}
</style>
