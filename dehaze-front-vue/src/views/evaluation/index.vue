<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import ModelAPI from "@/api/model";
import { EvalResult } from "@/api/model/model";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import AlgorithmAPI from "@/api/algorithm";
import { Algorithm } from "@/api/algorithm/model";
import ParallelImageShow from "@/components/ParallelImageShow/index.vue";
import ParallelImageUpload from "@/components/ParallelImageUpload/index.vue";
import { Arrayable } from "@vueuse/core";

defineOptions({
  name: "Evaluation",
});

const imageShowStore = useImageShowStore();
const { modelId } = toRefs(imageShowStore);
const loading = ref(false);
const showResult = ref(false);
const state = reactive({
  magnifier: {
    enabled: imageShowStore.magnifierInfo.enabled,
    shape: imageShowStore.magnifierInfo.shape,
    width: imageShowStore.magnifierInfo.width,
    height: imageShowStore.magnifierInfo.height,
    zoomLevel: imageShowStore.magnifierInfo.zoomLevel,
  },
  brightness: {
    enabled: true,
    value: 0,
  },
  contrast: {
    enabled: true,
    value: 0,
  },
  saturate: {
    enabled: true,
    value: 0,
  },
});

const algorithmInfo = ref<Algorithm>({
  id: 0,
  parentId: 0,
  name: "未知",
  description: "未知",
} as Algorithm);
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
  // TODO 让图片全部重置
}

const allUploaded = computed(() => {
  return gt.value && pred.value && haze.value;
});

async function handleEvaluation() {
  if (!allUploaded.value) {
    ElMessage.error("请先上传图片");
  }
  loading.value = true;
  try {
    algorithmInfo.value = await AlgorithmAPI.getAlgorithmInfoById(
      modelId.value
    );

    metrics.value = await ModelAPI.evaluation({
      modelId: modelId.value,
      predUrl: pred.value.url,
      gtUrl: gt.value.url,
    });
    showResult.value = true;
  } catch (e) {
    console.log(e);
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  if (allUploaded.value) {
    await handleEvaluation();
  }
});
</script>

<template>
  <div class="app-container">
    <el-card>
      <div class="evaluation-header">
        <div></div>
        <div class="title">图像效果评估</div>
        <el-popover :width="400" placement="bottom-start" trigger="click">
          <template #reference>
            <div class="settings">
              <i-ep-setting class="icon" />
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

      <ParallelImageShow v-if="showResult" />

      <div v-else>
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

      <div v-if="showResult" class="flex">
        <div style="min-width: 42vw; padding-right: 20px">
          <h3 class="text-center">算法说明</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item :span="2" :width="120" label="算法名称">
              {{ algorithmInfo.name }}
            </el-descriptions-item>
            <el-descriptions-item label="类型"
              >{{ algorithmInfo.type }}
            </el-descriptions-item>
            <el-descriptions-item label="权重大小">
              {{ algorithmInfo.size }}
            </el-descriptions-item>
            <el-descriptions-item v-if="algorithmInfo.flops" label="浮点数量">
              {{ algorithmInfo.flops }}
            </el-descriptions-item>
            <el-descriptions-item v-if="algorithmInfo.params" label="参数量">
              {{ algorithmInfo.params }}
            </el-descriptions-item>
            <el-descriptions-item :span="2" label="算法描述">
              {{ algorithmInfo.description }}
            </el-descriptions-item>
            <el-descriptions-item label="网络架构">
              <div style="height: 105px"></div>
            </el-descriptions-item>
          </el-descriptions>
        </div>

        <div style="min-width: 50vw; padding-left: 20px">
          <h3 class="text-center">指标评价</h3>
          <el-table :data="metrics">
            <el-table-column :width="90" fixed label="指标" prop="label" />
            <el-table-column :width="125" align="center" label="值">
              <template #default="scope">
                <span>{{ scope.row.value.toFixed(4) }}&nbsp;&nbsp;</span>

                <span v-if="scope.row.better === 'higher'">
                  <el-tag type="success"> ↑ </el-tag>
                </span>
                <span v-else-if="scope.row.better === 'lower'">
                  <el-tag type="danger"> ↓ </el-tag>
                </span>
              </template>
            </el-table-column>
            <el-table-column :min-width="300" label="描述" prop="description" />
          </el-table>
        </div>
      </div>

      <el-empty v-else description="暂无内容" />
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

<style lang="scss">
.el-alert {
  margin-bottom: 16px;
}
</style>
