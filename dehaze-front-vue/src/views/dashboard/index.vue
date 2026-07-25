<template>
  <div class="dashboard-container">
    <!-- 欢迎横幅 -->
    <el-card shadow="never" class="welcome-card">
      <div class="welcome-inner">
        <div class="welcome-left">
          <img :src="avatarUrl" class="user-avatar" />
          <div class="welcome-text">
            <h2 class="greeting">{{ greetings }}</h2>
            <p class="subtitle">图像去雾智能处理平台 · 让每一帧画面恢复清晰</p>
          </div>
        </div>
        <div class="welcome-right">
          <div class="quick-entry" @click="router.push('/presentation/dehaze')">
            <el-icon class="entry-icon"><MagicStick /></el-icon>
            <span>开始去雾</span>
          </div>
          <div class="quick-entry" @click="router.push('/dataset/list')">
            <el-icon class="entry-icon"><Files /></el-icon>
            <span>数据集</span>
          </div>
          <div class="quick-entry" @click="router.push('/algorithm/list')">
            <el-icon class="entry-icon"><Cpu /></el-icon>
            <span>算法库</span>
          </div>
        </div>
      </div>
    </el-card>

    <!-- 核心指标卡片 -->
    <el-row :gutter="16" class="stat-row">
      <el-col
        v-for="item in statCards"
        :key="item.key"
        :xs="12"
        :sm="12"
        :lg="6"
      >
        <el-card
          shadow="hover"
          class="stat-card"
          @click="router.push(item.link)"
        >
          <div class="stat-card-inner">
            <div class="stat-icon-wrap" :style="{ background: item.bg }">
              <el-icon class="stat-icon"><component :is="item.icon" /></el-icon>
            </div>
            <div class="stat-info">
              <div class="stat-value">{{ item.value }}</div>
              <div class="stat-label">{{ item.label }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 图表区 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :xs="24" :lg="16">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header">
              <span class="header-title">
                <el-icon class="header-icon"><TrendCharts /></el-icon
                >近7天任务处理趋势
              </span>
              <el-tag type="info" size="small" effect="plain">单位：次</el-tag>
            </div>
          </template>
          <div ref="trendChartRef" class="chart-box"></div>
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="8">
        <el-card shadow="never" class="chart-card">
          <template #header>
            <div class="card-header">
              <span class="header-title">
                <el-icon class="header-icon"><PieChart /></el-icon>任务状态分布
              </span>
            </div>
          </template>
          <div ref="pieChartRef" class="chart-box"></div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 最近任务 -->
    <el-card shadow="never" class="recent-card">
      <template #header>
        <div class="card-header">
          <span class="header-title">
            <el-icon class="header-icon"><Clock /></el-icon>最近任务
          </span>
          <el-button text type="primary" @click="router.push('/task/list')">
            查看全部<el-icon><ArrowRight /></el-icon>
          </el-button>
        </div>
      </template>
      <el-table
        :data="recentTasks"
        v-loading="taskLoading"
        empty-text="暂无任务记录"
      >
        <el-table-column
          label="任务ID"
          prop="taskId"
          width="280"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="120" align="center">
          <template #default="{ row }">{{
            taskTypeLabel[row.taskType] ?? row.taskType
          }}</template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag
              :color="statusTagColor[row.status]"
              effect="dark"
              size="small"
            >
              {{ statusLabel[row.status] }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="200">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="progressStatus(row.status)"
              :stroke-width="12"
              :text-inside="true"
            />
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="180" align="center">
          <template #default="{ row }">{{
            formatTime(row.createdAt)
          }}</template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script lang="ts" setup>
import { useUserStore } from "@/store/modules/user";
import { useTaskStore } from "@/store/modules/task";
import { AlgorithmAPI, DatasetAPI } from "dehaze-sdk-js";
import * as echarts from "echarts";
import {
  FolderOpened,
  Cpu,
  List,
  CircleCheck,
  MagicStick,
  Files,
  TrendCharts,
  PieChart,
  Clock,
  ArrowRight,
} from "@element-plus/icons-vue";

defineOptions({
  name: "Dashboard",
  inheritAttrs: false,
});

const router = useRouter();
const userStore = useUserStore();
const taskStore = useTaskStore();

// 头像 URL：仅对 HTTP(S) 链接追加图片处理参数，data: URL（如 base64）保持原样
const avatarUrl = computed(() => {
  const avatar = userStore.user.avatar || "";
  if (!avatar) return "";
  if (/^https?:\/\//i.test(avatar)) {
    return avatar + "?imageView2/1/w/80/h/80";
  }
  return avatar;
});

const greetings = computed(() => {
  const h = new Date().getHours();
  const name = userStore.user.nickname || "管理员";
  if (h >= 6 && h < 12) return `早上好，${name}！`;
  if (h >= 12 && h < 18) return `下午好，${name}！`;
  if (h >= 18 && h < 24) return `晚上好，${name}！`;
  return `夜深了，${name}，注意休息🌙`;
});

// 核心指标
const datasetCount = ref(0);
const algorithmCount = ref(0);
const taskTotal = ref(0);
const completedTaskCount = ref(0);

const statCards = computed(() => [
  {
    key: "dataset",
    label: "数据集总数",
    value: datasetCount.value,
    icon: FolderOpened,
    bg: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    link: "/dataset/list",
  },
  {
    key: "algorithm",
    label: "可用算法",
    value: algorithmCount.value,
    icon: Cpu,
    bg: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    link: "/algorithm/list",
  },
  {
    key: "task",
    label: "任务总数",
    value: taskTotal.value,
    icon: List,
    bg: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    link: "/task/list",
  },
  {
    key: "completed",
    label: "已完成任务",
    value: completedTaskCount.value,
    icon: CircleCheck,
    bg: "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    link: "/task/list",
  },
]);

// 最近任务
const recentTasks = ref<any[]>([]);
const taskLoading = ref(false);

const taskTypeLabel: Record<string, string> = {
  DEHAZE: "图像去雾",
  BATCH_DEHAZE: "批量去雾",
  EVALUATION: "算法评测",
};
const statusLabel: Record<string, string> = {
  PENDING: "待执行",
  PROCESSING: "执行中",
  COMPLETED: "已完成",
  FAILED: "失败",
  CANCELLED: "已取消",
};
const statusTagColor: Record<string, string> = {
  PENDING: "#1890ff",
  PROCESSING: "#1890ff",
  COMPLETED: "#52c41a",
  FAILED: "#ff4d4f",
  CANCELLED: "#8c8c8c",
};

function progressStatus(status: string) {
  if (status === "COMPLETED") return "success";
  if (status === "FAILED") return "exception";
  return undefined;
}

function formatTime(t?: string) {
  if (!t) return "-";
  return new Date(t).toLocaleString("zh-CN", { hour12: false });
}

// 加载统计数据
async function loadStats() {
  try {
    const [dsRes, algRes, taskRes] = await Promise.all([
      DatasetAPI.getList({ pageNum: 1, pageSize: 1 }),
      AlgorithmAPI.getList({}),
      taskStore.getTaskList({ pageNum: 1, pageSize: 5 }),
    ]);
    datasetCount.value = dsRes.total || 0;
    algorithmCount.value = Array.isArray(algRes) ? algRes.length : 0;
    taskTotal.value = taskStore.total || 0;
    recentTasks.value = taskStore.taskList.slice(0, 5) || [];
    // 统计已完成
    completedTaskCount.value = recentTasks.value.filter(
      (t) => t.status === "COMPLETED"
    ).length;
    // 注意：这是最近5条的已完成数，仅作展示用，精确值需要单独接口
  } catch (e) {
    // 静默处理，统计失败不影响首页渲染
  }
}

// 趋势图
const trendChartRef = ref<HTMLDivElement>();
const pieChartRef = ref<HTMLDivElement>();
let trendChart: echarts.ECharts | null = null;
let pieChart: echarts.ECharts | null = null;

function initTrendChart() {
  if (!trendChartRef.value) return;
  trendChart = echarts.init(trendChartRef.value);
  // 生成最近7天日期
  const dates: string[] = [];
  const values: number[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(Date.now() - i * 24 * 3600 * 1000);
    dates.push(
      `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`
    );
    // 模拟任务数据（实际可由后端统计接口提供）
    values.push(Math.floor(Math.random() * 30) + 10);
  }
  trendChart.setOption({
    tooltip: { trigger: "axis" },
    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
    xAxis: {
      type: "category",
      data: dates,
      boundaryGap: false,
      axisLine: { lineStyle: { color: "#dcdfe6" } },
      axisLabel: { color: "#606266" },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: "#f0f0f0" } },
      axisLabel: { color: "#606266" },
    },
    series: [
      {
        name: "任务数",
        type: "line",
        smooth: true,
        data: values,
        symbol: "circle",
        symbolSize: 8,
        lineStyle: { width: 3, color: "#409eff" },
        itemStyle: { color: "#409eff" },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: "rgba(64,158,255,0.5)" },
            { offset: 1, color: "rgba(64,158,255,0)" },
          ]),
        },
      },
    ],
  });
}

function initPieChart() {
  if (!pieChartRef.value) return;
  pieChart = echarts.init(pieChartRef.value);
  pieChart.setOption({
    tooltip: { trigger: "item", formatter: "{b}: {c} ({d}%)" },
    legend: {
      bottom: 0,
      icon: "circle",
      textStyle: { color: "#606266" },
    },
    color: ["#67c23a", "#e6a23c", "#409eff", "#f56c6c", "#909399"],
    series: [
      {
        type: "pie",
        radius: ["45%", "70%"],
        center: ["50%", "45%"],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 8,
          borderColor: "#fff",
          borderWidth: 2,
        },
        label: { show: false, position: "center" },
        emphasis: {
          label: { show: true, fontSize: 18, fontWeight: "bold" },
        },
        labelLine: { show: false },
        data: [
          { value: 0, name: "已完成" },
          { value: 0, name: "执行中" },
          { value: 0, name: "待执行" },
          { value: 0, name: "失败" },
          { value: 0, name: "已取消" },
        ],
      },
    ],
  });
}

// 根据最近任务数据更新饼图
function updatePieChart(tasks: any[]) {
  if (!pieChart) return;
  const counts: Record<string, number> = {
    COMPLETED: 0,
    PROCESSING: 0,
    PENDING: 0,
    FAILED: 0,
    CANCELLED: 0,
  };
  tasks.forEach((t) => {
    if (counts[t.status] !== undefined) counts[t.status]++;
  });
  pieChart.setOption({
    series: [
      {
        data: [
          { value: counts.COMPLETED, name: "已完成" },
          { value: counts.PROCESSING, name: "执行中" },
          { value: counts.PENDING, name: "待执行" },
          { value: counts.FAILED, name: "失败" },
          { value: counts.CANCELLED, name: "已取消" },
        ],
      },
    ],
  });
}

function handleResize() {
  trendChart?.resize();
  pieChart?.resize();
}

onMounted(async () => {
  await loadStats();
  initTrendChart();
  initPieChart();
  updatePieChart(taskStore.taskList);
  window.addEventListener("resize", handleResize);
});

onActivated(() => {
  handleResize();
});

onUnmounted(() => {
  window.removeEventListener("resize", handleResize);
  trendChart?.dispose();
  pieChart?.dispose();
});
</script>

<style lang="scss" scoped>
.dashboard-container {
  min-height: calc(100vh - var(--navbar-height));
  padding: 16px;
  background: var(--el-bg-color-page);
}

.welcome-card {
  margin-bottom: 16px;
  color: #fff;
  background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 12px;

  :deep(.el-card__body) {
    padding: 24px 28px;
  }

  .welcome-inner {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
    justify-content: space-between;
  }

  .welcome-left {
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .user-avatar {
    width: 64px;
    height: 64px;
    object-fit: cover;
    background: rgb(255 255 255 / 10%);
    border: 3px solid rgb(255 255 255 / 40%);
    border-radius: 50%;
  }

  .greeting {
    margin: 0 0 6px;
    font-size: 20px;
    font-weight: 600;
    color: #fff;
  }

  .subtitle {
    margin: 0;
    font-size: 13px;
    color: rgb(255 255 255 / 80%);
  }

  .welcome-right {
    display: flex;
    gap: 12px;
  }

  .quick-entry {
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: center;
    padding: 12px 16px;
    font-size: 12px;
    color: #fff;
    cursor: pointer;
    background: rgb(255 255 255 / 15%);
    border: 1px solid rgb(255 255 255 / 20%);
    border-radius: 10px;
    transition: all 0.25s;

    .entry-icon {
      font-size: 22px;
    }

    &:hover {
      background: rgb(255 255 255 / 25%);
      transform: translateY(-2px);
    }
  }
}

.stat-row {
  margin-bottom: 16px;
}

.stat-card {
  margin-bottom: 12px;
  cursor: pointer;
  border: none;
  border-radius: 12px;
  transition: all 0.25s;

  &:hover {
    box-shadow: 0 8px 20px rgb(0 0 0 / 8%);
    transform: translateY(-3px);
  }

  :deep(.el-card__body) {
    padding: 20px;
  }

  .stat-card-inner {
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .stat-icon-wrap {
    display: flex;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    border-radius: 12px;

    .stat-icon {
      font-size: 28px;
      color: #fff;
    }
  }

  .stat-info {
    flex: 1;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.2;
    color: var(--el-text-color-primary);
  }

  .stat-label {
    margin-top: 4px;
    font-size: 13px;
    color: var(--el-text-color-secondary);
  }
}

.chart-row {
  margin-bottom: 16px;
}

.chart-card {
  height: 100%;
  border: none;
  border-radius: 12px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    .header-title {
      display: flex;
      gap: 6px;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .header-icon {
      font-size: 18px;
      color: var(--el-color-primary);
    }
  }

  .chart-box {
    width: 100%;
    height: 320px;
  }
}

.recent-card {
  border: none;
  border-radius: 12px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    .header-title {
      display: flex;
      gap: 6px;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .header-icon {
      font-size: 18px;
      color: var(--el-color-primary);
    }
  }
}

@media (width <= 768px) {
  .welcome-card .welcome-inner {
    flex-direction: column;
    align-items: flex-start;
  }

  .welcome-right {
    justify-content: space-between;
    width: 100%;
  }
}
</style>
