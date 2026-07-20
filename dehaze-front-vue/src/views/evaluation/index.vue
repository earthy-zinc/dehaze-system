<script lang="ts" setup>
import ParallelImageShow from "@/components/ParallelImageShow/index.vue";
import ParallelImageUpload from "@/components/ParallelImageUpload/index.vue";

import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useImageShowStore } from "@/store/modules/imageShow";
import { Arrayable } from "@vueuse/core";
import { Algorithm, AlgorithmAPI, EvalResult, ModelAPI } from "dehaze-sdk-js";
import { Setting } from "@element-plus/icons-vue";
import * as echarts from "echarts";

defineOptions({
  name: "Evaluation",
});

const imageShowStore = useImageShowStore();
const { modelId, dehazeParams } = toRefs(imageShowStore);
const loading = ref(false);
const showResult = ref(false);
// 对比模式：指标对比 / 图片对比 / 参数对比
const activeTab = ref<"metrics" | "image" | "params">("metrics");

const state = reactive({
  magnifier: {
    enabled: imageShowStore.magnifierInfo.enabled,
    shape: imageShowStore.magnifierInfo.shape,
    width: imageShowStore.magnifierInfo.width,
    height: imageShowStore.magnifierInfo.height,
    zoomLevel: imageShowStore.magnifierInfo.zoomLevel,
  },
  brightness: {
    value: 0,
  },
  contrast: {
    value: 0,
  },
  saturate: {
    value: 0,
  },
});

const algorithmInfo = ref<Algorithm | null>(null);
const metrics = ref<EvalResult[]>();

const { imageInfo } = toRefs(imageShowStore);

const pred = computed(
  () =>
    imageInfo.value.images.urls.filter(
      (img) => img.label.text === ImageTypeEnum.PRED
    )[0]
);
const gt = computed(
  () =>
    imageInfo.value.images.urls.filter(
      (img) => img.label.text === ImageTypeEnum.CLEAN
    )[0]
);
const haze = computed(
  () =>
    imageInfo.value.images.urls.filter(
      (img) => img.label.text === ImageTypeEnum.HAZE
    )[0]
);

function transform(x: number) {
  return 0.5 * x + 100;
}

function handleMagnifierChange(
  value: any,
  type: "shape" | "zoomLevel" | "width" | "height" | "enable"
) {
  switch (type) {
    case "enable":
      state.magnifier.enabled = !state.magnifier.enabled;
      imageShowStore.setMagnifierShow(state.magnifier.enabled);
      break;
    case "shape":
      imageShowStore.setMagnifierShape(value);
      break;
    case "zoomLevel":
      imageShowStore.setMagnifierZoomLevel(value);
      break;
    case "height":
      imageShowStore.setMagnifierSize(state.magnifier.width, value);
      break;
    case "width":
      imageShowStore.setMagnifierSize(value, state.magnifier.height);
      break;
    default:
      break;
  }
}

function handleImageFilterChange(
  value: number,
  type: "brightness" | "contrast" | "saturate"
) {
  value = transform(value);
  switch (type) {
    case "brightness":
      imageShowStore.setBrightness(value);
      break;
    case "contrast":
      imageShowStore.setContrast(value);
      break;
    case "saturate":
      imageShowStore.setSaturate(value);
      break;
    default:
      break;
  }
}

function handleReset() {
  imageShowStore.setImageUrls([]);
  showResult.value = false;
  metrics.value = undefined;
  algorithmInfo.value = null;
}

const allUploaded = computed(() => {
  return gt.value && pred.value && haze.value;
});

async function handleEvaluation() {
  if (!allUploaded.value) {
    ElMessage.error("请先上传图片");
    return;
  }
  loading.value = true;
  try {
    algorithmInfo.value = await AlgorithmAPI.getAlgorithmInfoById(
      modelId.value
    );

    metrics.value = await ModelAPI.evaluation({
      modelId: modelId.value,
      predUrl: pred.value!.url,
      gtUrl: gt.value!.url,
    });
    showResult.value = true;
  } catch (e: any) {
    ElMessage.error("评估失败：" + (e.message || "未知错误"));
  } finally {
    loading.value = false;
  }
}

// ============ 指标可视化图表（雷达图 + 柱状图） ============
const radarChartRef = ref<HTMLDivElement>();
const barChartRef = ref<HTMLDivElement>();
let radarChart: echarts.ECharts | null = null;
let barChart: echarts.ECharts | null = null;

// 雷达图指标配置（每个指标独立的最大值，使各维度可比较）
const radarIndicators = computed(() => {
  if (!metrics.value) return [];
  return metrics.value.map((m) => {
    const v = Number(m.value);
    const baseline = m.baseline ? Number(m.baseline) : 0;
    // 取值与基准的较大者作为参考，再留出 30% 余量
    const ref = Math.max(v, baseline, 1);
    return { name: m.label, max: ref * 1.3 };
  });
});

function renderRadarChart() {
  if (!radarChartRef.value || !metrics.value || metrics.value.length === 0)
    return;
  if (!radarChart) {
    radarChart = markRaw(echarts.init(radarChartRef.value));
  }
  const currentValues = metrics.value.map((m) => Number(m.value));
  const baselineValues = metrics.value.map((m) =>
    m.baseline ? Number(m.baseline) : 0
  );
  const hasBaseline = metrics.value.some((m) => m.baseline);
  const series: any[] = [
    {
      value: currentValues,
      name: "本次评估",
      areaStyle: { opacity: 0.2 },
    },
  ];
  if (hasBaseline) {
    series.push({
      value: baselineValues,
      name: "基准值",
      areaStyle: { opacity: 0.1 },
    });
  }
  radarChart.setOption({
    tooltip: {},
    legend: {
      data: hasBaseline ? ["本次评估", "基准值"] : ["本次评估"],
      bottom: 0,
    },
    radar: {
      indicator: radarIndicators.value,
      radius: "65%",
    },
    series: [
      {
        type: "radar",
        data: series,
      },
    ],
  });
}

function renderBarChart() {
  if (!barChartRef.value || !metrics.value || metrics.value.length === 0)
    return;
  if (!barChart) {
    barChart = markRaw(echarts.init(barChartRef.value));
  }
  const labels = metrics.value.map((m) => m.label);
  const values = metrics.value.map((m) => Number(m.value));
  const baselines = metrics.value.map((m) =>
    m.baseline ? Number(m.baseline) : 0
  );
  const hasBaseline = metrics.value.some((m) => m.baseline);
  const series: any[] = [
    {
      name: "本次评估",
      type: "bar",
      data: values,
      itemStyle: { color: "#409EFF" },
    },
  ];
  if (hasBaseline) {
    series.push({
      name: "基准值",
      type: "bar",
      data: baselines,
      itemStyle: { color: "#E6A23C" },
    });
  }
  barChart.setOption({
    tooltip: { trigger: "axis" },
    legend: {
      data: hasBaseline ? ["本次评估", "基准值"] : ["本次评估"],
      bottom: 0,
    },
    grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: { interval: 0, rotate: 0 },
    },
    yAxis: { type: "value" },
    series,
  });
}

// 监听指标数据变化，渲染图表
watch(
  metrics,
  () => {
    nextTick(() => {
      renderRadarChart();
      renderBarChart();
    });
  },
  { deep: true }
);

// 切换到指标对比 Tab 时重新调整图表尺寸
watch(activeTab, (val) => {
  if (val === "metrics") {
    nextTick(() => {
      radarChart?.resize();
      barChart?.resize();
    });
  }
});

function handleWindowResize() {
  radarChart?.resize();
  barChart?.resize();
}

// ============ 参数对比数据 ============
// 处理参数 vs 默认参数
const paramCompareData = computed(() => [
  {
    name: "去雾强度",
    current: dehazeParams.value.dehazeStrength,
    default: 50,
  },
  {
    name: "色彩饱和度",
    current: dehazeParams.value.colorSaturation,
    default: 50,
  },
  {
    name: "对比度",
    current: dehazeParams.value.contrast,
    default: 50,
  },
  {
    name: "锐化程度",
    current: dehazeParams.value.sharpen,
    default: 30,
  },
]);

// ============ 导出报告 ============
function handleExportReport() {
  if (!metrics.value || metrics.value.length === 0) {
    ElMessage.error("暂无评估结果可导出");
    return;
  }
  const algo = algorithmInfo.value;
  let content = "图像去雾效果评估报告\n";
  content += "========================================\n";
  content += `生成时间：${new Date().toLocaleString()}\n`;
  content += `算法名称：${algo?.name ?? "未知"}\n`;
  content += `算法类型：${algo?.type ?? "未知"}\n`;
  content += `算法描述：${algo?.description ?? ""}\n`;
  content += "========================================\n";
  content += "一、评估指标\n";
  metrics.value.forEach((m) => {
    content += `  ${m.label}：${Number(m.value).toFixed(4)}`;
    if (m.better === "higher") content += "（越高越好）";
    else if (m.better === "lower") content += "（越低越好）";
    if (m.baseline) content += `，基准值：${m.baseline}`;
    content += "\n";
  });
  content += "========================================\n";
  content += "二、处理参数\n";
  content += `  去雾强度：${dehazeParams.value.dehazeStrength}（默认 50）\n`;
  content += `  色彩饱和度：${dehazeParams.value.colorSaturation}（默认 50）\n`;
  content += `  对比度：${dehazeParams.value.contrast}（默认 50）\n`;
  content += `  锐化程度：${dehazeParams.value.sharpen}（默认 30）\n`;
  content += "========================================\n";

  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `evaluation_report_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  ElMessage.success("报告已导出");
}

onMounted(async () => {
  window.addEventListener("resize", handleWindowResize);
  if (allUploaded.value) {
    await handleEvaluation();
  }
});

onUnmounted(() => {
  window.removeEventListener("resize", handleWindowResize);
  radarChart?.dispose();
  barChart?.dispose();
});
</script>

<template>
  <div class="app-container">
    <el-card>
      <div class="evaluation-header">
        <el-button
          :disabled="!showResult"
          type="primary"
          @click="handleExportReport"
        >
          导出报告
        </el-button>
        <div class="title">图像效果评估</div>
        <el-popover :width="400" placement="bottom-start" trigger="click">
          <template #reference>
            <div class="settings">
              <el-icon class="icon"><Setting /></el-icon>
            </div>
          </template>
          <template #default>
            <h3 style="margin-top: 8px; font-size: 1.25rem; text-align: center">
              图像对比工具
            </h3>
            <el-divider style="margin-top: -4px; margin-bottom: 18px" />
            <el-form>
              <el-form-item class="more-operations" label="放大镜形状">
                <el-radio-group
                  v-model="state.magnifier.shape"
                  @change="
                    (value: string | number | boolean | undefined) =>
                      handleMagnifierChange(value, 'shape')
                  "
                >
                  <el-radio label="square" value="square">正方形</el-radio>
                  <el-radio label="circle" value="circle">圆形</el-radio>
                </el-radio-group>
              </el-form-item>

              <el-form-item class="more-operations" label="放大倍数">
                <el-slider
                  v-model="state.magnifier.zoomLevel"
                  :max="20"
                  :min="2"
                  @change="
                    (value: Arrayable<number>) =>
                      handleMagnifierChange(value, 'zoomLevel')
                  "
                />
              </el-form-item>

              <el-form-item class="more-operations" label="放大镜宽度">
                <el-slider
                  v-model="state.magnifier.width"
                  :max="1000"
                  :min="100"
                  @change="
                    (value: Arrayable<number>) =>
                      handleMagnifierChange(value, 'width')
                  "
                />
              </el-form-item>
              <el-form-item class="more-operations" label="放大镜高度">
                <el-slider
                  v-model="state.magnifier.height"
                  :max="1000"
                  :min="100"
                  @change="
                    (value: Arrayable<number>) =>
                      handleMagnifierChange(value, 'height')
                  "
                />
              </el-form-item>
              <el-form-item class="more-operations" label="亮度">
                <el-slider
                  v-model="state.brightness.value"
                  :max="100"
                  :min="-100"
                  @change="
                    (value: Arrayable<number>) =>
                      handleImageFilterChange(Number(value), 'brightness')
                  "
                />
              </el-form-item>
              <el-form-item class="more-operations" label="对比度">
                <el-slider
                  v-model="state.contrast.value"
                  :max="100"
                  :min="-100"
                  @change="
                    (value: Arrayable<number>) =>
                      handleImageFilterChange(Number(value), 'contrast')
                  "
                />
              </el-form-item>
              <el-form-item class="more-operations" label="饱和度">
                <el-slider
                  v-model="state.saturate.value"
                  :max="100"
                  :min="-100"
                  @change="
                    (value: Arrayable<number>) =>
                      handleImageFilterChange(Number(value), 'saturate')
                  "
                />
              </el-form-item>
            </el-form>
          </template>
        </el-popover>
      </div>

      <el-alert
        v-if="!showResult"
        description="全部图像上传完毕后开始评估"
        show-icon
        type="warning"
      />

      <!-- 评估前：上传图片 -->
      <div v-if="!showResult && !loading">
        <ParallelImageUpload />
        <div class="flex justify-center mt-6">
          <el-button size="large" @click="handleReset">重新上传</el-button>
          <el-button
            :disabled="!allUploaded"
            :loading="loading"
            size="large"
            type="primary"
            @click="handleEvaluation"
            >开始评估
          </el-button>
        </div>
      </div>

      <el-skeleton v-if="loading" :rows="10" animated class="mt-10" />

      <!-- 评估后：对比模式切换 -->
      <div v-if="showResult">
        <el-tabs v-model="activeTab" class="eval-tabs">
          <!-- 指标对比：表格 + 可视化图表 -->
          <el-tab-pane label="指标对比" name="metrics">
            <div class="flex metrics-row">
              <div style="min-width: 42vw; padding-right: 20px">
                <h3 class="text-center">算法说明</h3>
                <el-descriptions :column="2" border>
                  <el-descriptions-item :span="2" :width="120" label="算法名称">
                    {{ algorithmInfo?.name }}
                  </el-descriptions-item>
                  <el-descriptions-item label="类型"
                    >{{ algorithmInfo?.type }}
                  </el-descriptions-item>
                  <el-descriptions-item label="权重大小">
                    {{ algorithmInfo?.size }}
                  </el-descriptions-item>
                  <el-descriptions-item
                    v-if="algorithmInfo?.flops"
                    label="浮点数量"
                  >
                    {{ algorithmInfo.flops }}
                  </el-descriptions-item>
                  <el-descriptions-item
                    v-if="algorithmInfo?.params"
                    label="参数量"
                  >
                    {{ algorithmInfo.params }}
                  </el-descriptions-item>
                  <el-descriptions-item :span="2" label="算法描述">
                    {{ algorithmInfo?.description }}
                  </el-descriptions-item>
                  <el-descriptions-item label="网络架构">
                    <div style="height: 105px"></div>
                  </el-descriptions-item>
                </el-descriptions>
              </div>

              <div style="min-width: 50vw; padding-left: 20px">
                <h3 class="text-center">指标评价</h3>
                <el-table :data="metrics">
                  <el-table-column
                    :width="90"
                    fixed
                    label="指标"
                    prop="label"
                  />
                  <el-table-column :width="125" align="center" label="值">
                    <template #default="scope">
                      <span
                        >{{
                          Number(scope.row.value).toFixed(4)
                        }}&nbsp;&nbsp;</span
                      >

                      <span v-if="scope.row.better === 'higher'">
                        <el-tag type="success"> ↑ </el-tag>
                      </span>
                      <span v-else-if="scope.row.better === 'lower'">
                        <el-tag type="danger"> ↓ </el-tag>
                      </span>
                    </template>
                  </el-table-column>
                  <el-table-column
                    :min-width="300"
                    label="描述"
                    prop="description"
                  />
                </el-table>
              </div>
            </div>

            <!-- 指标可视化图表 -->
            <el-divider content-position="center">指标可视化</el-divider>
            <div class="charts-wrap">
              <div class="chart-item">
                <h4 class="text-center">雷达图（多维度指标对比）</h4>
                <div ref="radarChartRef" class="chart-canvas"></div>
              </div>
              <div class="chart-item">
                <h4 class="text-center">柱状图（各指标值）</h4>
                <div ref="barChartRef" class="chart-canvas"></div>
              </div>
            </div>
          </el-tab-pane>

          <!-- 图片对比：并排展示 -->
          <el-tab-pane label="图片对比" name="image">
            <ParallelImageShow />
          </el-tab-pane>

          <!-- 参数对比：处理参数 vs 默认参数 -->
          <el-tab-pane label="参数对比" name="params">
            <h3 class="text-center">处理参数对比</h3>
            <el-table :data="paramCompareData" border>
              <el-table-column label="参数名称" prop="name" />
              <el-table-column align="center" label="本次处理值">
                <template #default="scope">
                  <span>{{ scope.row.current }}</span>
                </template>
              </el-table-column>
              <el-table-column align="center" label="默认值">
                <template #default="scope">
                  <span>{{ scope.row.default }}</span>
                </template>
              </el-table-column>
              <el-table-column label="差异">
                <template #default="scope">
                  <el-tag
                    :type="
                      scope.row.current === scope.row.default
                        ? 'info'
                        : 'warning'
                    "
                  >
                    {{
                      scope.row.current === scope.row.default
                        ? "一致"
                        : "已调整（差值 " +
                          (scope.row.current - scope.row.default) +
                          "）"
                    }}
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </div>

      <el-empty v-else-if="!loading" description="暂无内容" />
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.evaluation-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 3rem 12px;
}

.title {
  font-size: 1.5rem;
  font-weight: bold;
}

.settings {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3rem;
  height: 3rem;
  font-size: 1.2rem;
  cursor: pointer;
  border-radius: 25%;

  &:hover {
    background-color: #f3f3f3;
  }

  &:active {
    color: green;
  }

  & .icon:hover {
    animation: rotate360 1s linear forwards;
  }
}

@keyframes rotate360 {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

.eval-tabs {
  margin-top: 8px;
}

.metrics-row {
  flex-wrap: wrap;
  align-items: flex-start;
}

.charts-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: space-around;
  margin-top: 16px;

  .chart-item {
    flex: 1;
    min-width: 360px;

    h4 {
      margin-bottom: 12px;
    }

    .chart-canvas {
      width: 100%;
      height: 360px;
    }
  }
}
</style>

<style lang="scss">
.el-alert {
  margin-bottom: 16px;
}
</style>
