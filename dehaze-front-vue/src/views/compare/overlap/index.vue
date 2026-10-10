<script lang="ts" setup>
import ParallelImageShow from "@/components/ParallelImageShow/index.vue";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useImageShowStore } from "@/store/modules/imageShow";
import { Arrayable } from "@vueuse/core";
import { AiModelAPI, AiModelVO, ModelAPI } from "dehaze-sdk-js";

interface MetricItem {
  label: string;
  value: number | string;
}
import { Setting, Download } from "@element-plus/icons-vue";

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

const modelInfo = ref<AiModelVO | null>(null);
const metrics = ref<MetricItem[]>();

// Export report（logId 为去雾处理的预测记录ID，来自 imageShowStore.predLogId）
const reportGenerating = ref(false);

async function handleExportReport() {
  const logId = imageShowStore.predLogId;
  if (!logId) {
    ElMessage.warning("当前没有可导出的对比记录");
    return;
  }
  reportGenerating.value = true;
  try {
    const res = await ModelAPI.generateReport({ logId });
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
          link.download = "dehaze-report.html";
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
  } catch (e: any) {
    // axios 类错误（HTTP/业务码）由全局钩子提示，这里只兜底轮询中本地抛出的生成失败
    if (!e.isAxiosError) {
      ElMessage.error("导出报告失败：" + (e.message || "未知错误"));
    }
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
  const currentModelId = modelId.value;
  if (!currentModelId) {
    ElMessage.error("缺少本次处理使用的图像模型");
    return;
  }
  AiModelAPI.listEnabledModels("image_edit")
    .then((models) => {
      modelInfo.value =
        models.find((m) => m.modelId === currentModelId) ?? null;
    })
    .catch(() => {});

  ModelAPI.evaluateAndWait({
    modelId: currentModelId,
    predUrl: pred.value.url,
    gtUrl: gt.value.url,
  })
    .then((res) => {
      if (res.status === 3) {
        throw new Error(res.errorMessage || "评估失败");
      }
      metrics.value = Object.entries(res.metrics || {}).map(
        ([label, value]) => ({
          label,
          value,
        })
      );
    })
    .catch((e: any) => {
      // axios 类错误（HTTP/业务码）由全局钩子提示，这里只兜底本地抛出的评估失败
      if (!e.isAxiosError) {
        ElMessage.error("评估失败：" + (e.message || "未知错误"));
      }
    });
});
</script>

<template>
  <div class="app-container">
    <el-card>
      <div class="evaluation-header">
        <el-button
          type="primary"
          @click="handleExportReport"
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
          <h3 class="text-center">模型信息</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item :span="2" :width="120" label="模型名称">
              {{ modelInfo?.displayName }}
            </el-descriptions-item>
            <el-descriptions-item :span="2" label="模型标识">
              {{ modelInfo?.modelId }}
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
</style>
