<script lang="ts" setup>
import ParallelImageShow from "@/components/ParallelImageShow/index.vue";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useImageShowStore } from "@/store/modules/imageShow";
import { Arrayable } from "@vueuse/core";
import {
  Algorithm,
  AlgorithmAPI,
  ModelAPI,
  CompareReportForm,
} from "dehaze-sdk-js";

interface MetricItem {
  label: string;
  value: number | string;
}
import { Setting, Picture, Download } from "@element-plus/icons-vue";

const imageShowStore = useImageShowStore();

defineOptions({ name: "CompareOverlap" });
const { modelId } = toRefs(imageShowStore);
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
const algorithmLoading = ref(false);
const algorithmError = ref(false);
const metrics = ref<MetricItem[]>();

// Export report
const reportDialogVisible = ref(false);
const reportGenerating = ref(false);
const lastLogId = ref(0);
const reportForm = ref<CompareReportForm>({
  logId: 0,
  format: "pdf",
  includeMetrics: true,
  includeFilters: false,
});

function openReportDialog() {
  reportForm.value.logId = lastLogId.value;
  reportDialogVisible.value = true;
}

async function handleExportReport() {
  if (reportForm.value.logId === 0) {
    ElMessage.warning("当前没有可导出的对比记录");
    return;
  }
  reportGenerating.value = true;
  try {
    const res = await ModelAPI.generateReport(reportForm.value);
    if (!res.taskId) {
      throw new Error("未返回任务ID");
    }
    // Poll for completion (status: 1=PROCESSING, 2=COMPLETED, 3=FAILED)
    while (true) {
      const status = await ModelAPI.getReportStatus(res.taskId);
      if (status.status === 2) {
        if (status.downloadUrl) {
          const link = document.createElement("a");
          link.href = status.downloadUrl;
          link.download = `dehaze-report.${reportForm.value.format}`;
          link.click();
        } else {
          ElMessage.success("报告生成完成，请前往任务中心下载");
        }
        break;
      }
      if (status.status === 3) {
        throw new Error(status.errorMessage || "报告生成失败");
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    reportDialogVisible.value = false;
  } catch (e: any) {
    ElMessage.error("导出报告失败：" + (e.message || "未知错误"));
  } finally {
    reportGenerating.value = false;
  }
}

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
      imageShowStore.magnifierInfo.enabled = state.magnifier.enabled;
      break;
    case "shape":
      imageShowStore.magnifierInfo.shape = value;
      break;
    case "zoomLevel":
      imageShowStore.magnifierInfo.zoomLevel = value;
      break;
    case "height":
      imageShowStore.magnifierInfo.width = state.magnifier.width;
      imageShowStore.magnifierInfo.height = value;
      break;
    case "width":
      imageShowStore.magnifierInfo.width = value;
      imageShowStore.magnifierInfo.height = state.magnifier.height;
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
      imageShowStore.imageInfo.brightness = value;
      break;
    case "contrast":
      imageShowStore.imageInfo.contrast = value;
      break;
    case "saturate":
      imageShowStore.imageInfo.saturate = value;
      break;
    default:
      break;
  }
}

onMounted(() => {
  if (!pred.value || !gt.value) {
    ElMessage.error("不存在图像");
    return;
  }
  algorithmLoading.value = true;
  algorithmError.value = false;
  AlgorithmAPI.getAlgorithmInfoById(modelId.value)
    .then((res) => {
      algorithmInfo.value = res;
    })
    .catch((e: any) => {
      algorithmError.value = true;
      ElMessage.error("获取算法信息失败：" + (e.message || "未知错误"));
    })
    .finally(() => {
      algorithmLoading.value = false;
    });

  ModelAPI.evaluateAndWait({
    algorithmId: modelId.value,
    predUrl: pred.value.url,
    gtUrl: gt.value.url,
  })
    .then((res) => {
      if (res.status === 3) {
        throw new Error(res.errorMessage || "评估失败");
      }
      lastLogId.value = (res as any).logId || 0;
      metrics.value = Object.entries(res.metrics || {}).map(
        ([label, value]) => ({
          label,
          value,
        })
      );
    })
    .catch((e: any) => {
      ElMessage.error("评估失败：" + (e.message || "未知错误"));
    });
});
</script>

<template>
  <div class="app-container">
    <el-card>
      <div class="evaluation-header">
        <el-button
          type="primary"
          @click="openReportDialog"
          :loading="reportGenerating"
        >
          <el-icon><Download /></el-icon>
          导出对比报告
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

      <ParallelImageShow />

      <div class="flex">
        <div style="padding-right: 20px">
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
            <el-descriptions-item v-if="algorithmInfo?.flops" label="浮点数量">
              {{ algorithmInfo.flops }}
            </el-descriptions-item>
            <el-descriptions-item v-if="algorithmInfo?.params" label="参数量">
              {{ algorithmInfo.params }}
            </el-descriptions-item>
            <el-descriptions-item :span="2" label="算法描述">
              {{ algorithmInfo?.description }}
            </el-descriptions-item>
            <el-descriptions-item :span="2" label="网络架构">
              <div v-if="algorithmLoading" class="arch-loading">
                <el-skeleton :rows="3" animated />
              </div>
              <div v-else-if="algorithmError" class="arch-empty">
                <el-icon :size="32" color="#999"><Picture /></el-icon>
                <div>网络架构图不可用</div>
              </div>
              <el-image
                v-else
                :src="algorithmInfo?.img || ''"
                fit="contain"
                class="network-arch"
                preview-teleported
              >
                <template #error>
                  <div class="arch-empty">
                    <el-icon :size="32" color="#999"><Picture /></el-icon>
                    <div>网络架构图加载失败</div>
                  </div>
                </template>
              </el-image>
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <div style="min-width: 50vw; padding-left: 20px">
          <h3 class="text-center">指标评价</h3>
          <el-table :data="metrics">
            <el-table-column :width="90" fixed label="指标" prop="label" />
            <el-table-column :width="160" align="center" label="值">
              <template #default="scope">
                {{ Number(scope.row.value).toFixed(4) }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>

    <el-dialog v-model="reportDialogVisible" title="导出对比报告" width="420px">
      <el-form label-position="top">
        <el-form-item label="报告格式">
          <el-radio-group v-model="reportForm.format">
            <el-radio label="pdf">PDF</el-radio>
            <el-radio label="image">图片 (PNG)</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="包含内容">
          <el-checkbox v-model="reportForm.includeMetrics"
            >包含评价指标</el-checkbox
          >
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="reportDialogVisible = false">取消</el-button>
        <el-button
          type="primary"
          @click="handleExportReport"
          :loading="reportGenerating"
        >
          导出
        </el-button>
      </template>
    </el-dialog>
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

.arch-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 105px;
}

.arch-empty {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: center;
  justify-content: center;
  min-height: 105px;
  font-size: 14px;
  color: #999;
}

.network-arch {
  max-height: 200px;
  border-radius: 4px;
}
</style>
