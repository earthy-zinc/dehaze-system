<script lang="ts" setup>
import Magnifier from "@/components/Magnifier/newIndex.vue";
import { useImageShowStore } from "@/store/modules/imageShow";
import { useWindowSize } from "@vueuse/core";
import { UploadFile, UploadUserFile } from "element-plus";

defineOptions({
  name: "AlgorithmToolBar",
});

const props = defineProps({
  title: {
    type: String,
    default: "图像去雾",
  },
  description: {
    type: String,
    default:
      "通过对遭受雾气影响的图像进行相应处理，恢复图像原本的纹理结构和细节信息，进而提升图像的能见度",
  },
  disableMore: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(["onUpload", "onTakePhoto", "onReset", "onGenerate"]);

const overlapImageStore = useImageShowStore();

const { width } = useWindowSize();

const labelPosition = ref<"top" | "left" | "right">("left");
watch(
  () => width.value,
  (newValue) => {
    if (newValue >= 992) {
      labelPosition.value = "left";
    } else {
      labelPosition.value = "top";
    }
  }
);

const state = reactive({
  magnifier: {
    enabled: false,
    shape: "square",
    width: 100,
    height: 100,
    zoomLevel: 1,
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
  divider: true,
});

const cardRef = ref<HTMLDivElement>();
onMounted(() => {
  const { width: barWidth } = useElementBounding(cardRef.value);
  watch(
    () => barWidth,
    () => {
      if (barWidth.value > 0) {
        let value = (barWidth.value - 30 - 40 - 20) / 2;
        state.magnifier.width = value;
        state.magnifier.height = value;
        handleMagnifierChange(value, "width");
        handleMagnifierChange(value, "height");
      }
    },
    { deep: true }
  );
});

function transform(x: number) {
  return 0.5 * x + 100;
}

function handleMagnifierChange(
  value: any,
  type: "shape" | "zoomLevel" | "width" | "height" | "enable"
) {
  switch (type) {
    case "enable":
      overlapImageStore.toggleMagnifierShow();
      state.magnifier.enabled = !state.magnifier.enabled;
      break;
    case "shape":
      overlapImageStore.setMagnifierShape(value);
      break;
    case "zoomLevel":
      overlapImageStore.setMagnifierZoomLevel(value);
      break;
    case "height":
      overlapImageStore.setMagnifierSize(state.magnifier.width, value);
      break;
    case "width":
      overlapImageStore.setMagnifierSize(value, state.magnifier.height);
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
      overlapImageStore.setBrightness(value);
      break;
    case "contrast":
      overlapImageStore.setContrast(value);
      break;
    case "saturate":
      overlapImageStore.setSaturate(value);
      break;
    default:
      break;
  }
}

function handleDividerChange() {
  state.divider = !state.divider;
  overlapImageStore.toggleDividerShow();
}

function handleUploadChange(uploadFile: UploadFile) {
  emit("onUpload", uploadFile.raw);
}

function handleUploadExceed(files: File[], uploadFiles: UploadUserFile[]) {
  uploadFiles.shift();
  uploadFiles.push(files[0]);
  handleUploadChange(uploadFiles[0] as UploadFile);
}
</script>

<template>
  <div class="mr-3">
    <el-card ref="cardRef" class="sidebar-card">
      <h2 class="text-center m2">{{ title }}</h2>
      <el-text class="m2">
        {{ description }}
      </el-text>

      <div class="flex justify-evenly m-4">
        <el-dropdown split-button type="primary">
          <el-upload
            :auto-upload="false"
            :limit="1"
            :on-change="handleUploadChange"
            :on-exceed="handleUploadExceed"
            :show-file-list="false"
            accept="image/gif, image/jpeg, image/jpg, image/png, image/svg"
            action="#"
          >
            本地上传
          </el-upload>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item @click="emit('onTakePhoto')">
                拍照上传
              </el-dropdown-item>
              <el-dropdown-item> 从现有数据集中选择</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <el-button id="algorithm-reset-button" @click="emit('onReset')"
          >清除结果
        </el-button>
      </div>
      <slot></slot>
      <div class="flex justify-evenly m-4">
        <el-button
          id="algorithm-generate-button"
          type="primary"
          @click="emit('onGenerate')"
        >
          立即生成
        </el-button>
        <el-dropdown :disabled="disableMore">
          <el-button :disabled="disableMore">更多功能</el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item
                @click="() => handleMagnifierChange(0, 'enable')"
              >
                {{ state.magnifier.enabled ? "关闭" : "开启" }}放大镜
              </el-dropdown-item>
              <el-dropdown-item @click="handleDividerChange">
                {{ state.divider ? "关闭" : "开启" }}拖拽线
              </el-dropdown-item>
              <el-dropdown-item
                @click="state.brightness.enabled = !state.brightness.enabled"
              >
                {{ state.brightness.enabled ? "关闭" : "开启" }}亮度调整
              </el-dropdown-item>
              <el-dropdown-item
                @click="state.contrast.enabled = !state.contrast.enabled"
              >
                {{ state.contrast.enabled ? "关闭" : "开启" }}对比度调整
              </el-dropdown-item>
              <el-dropdown-item
                @click="state.saturate.enabled = !state.saturate.enabled"
              >
                {{ state.saturate.enabled ? "关闭" : "开启" }}饱和度调整
              </el-dropdown-item>
              <el-dropdown-item> 评估结果</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>

      <template v-if="state.magnifier.enabled && !disableMore">
        <div class="flex justify-between flex-wrap items-center mb-3">
          <Magnifier
            v-for="url in overlapImageStore.imageInfo.images.urls"
            :key="url.id"
            :label="url.label"
            :src="url.url"
          />
        </div>
      </template>

      <el-form :label-position="labelPosition" label-width="7.2vw">
        <template v-if="state.magnifier.enabled && !disableMore">
          <el-form-item class="more-operations" label="放大镜形状">
            <el-radio-group
              v-model="state.magnifier.shape"
              @change="(value) => handleMagnifierChange(value, 'shape')"
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
              @change="(value) => handleMagnifierChange(value, 'zoomLevel')"
            />
          </el-form-item>

          <el-form-item class="more-operations" label="放大镜宽度">
            <el-slider
              v-model="state.magnifier.width"
              :max="1000"
              :min="100"
              @change="(value) => handleMagnifierChange(value, 'width')"
            />
          </el-form-item>
          <el-form-item class="more-operations" label="放大镜高度">
            <el-slider
              v-model="state.magnifier.height"
              :max="1000"
              :min="100"
              @change="(value) => handleMagnifierChange(value, 'height')"
            />
          </el-form-item>
        </template>

        <template v-if="state.brightness.enabled && !disableMore">
          <el-form-item class="more-operations" label="亮度">
            <el-slider
              v-model="state.brightness.value"
              :max="100"
              :min="-100"
              @change="
                (value) => handleImageFilterChange(Number(value), 'brightness')
              "
            />
          </el-form-item>
        </template>

        <template v-if="state.contrast.enabled && !disableMore">
          <el-form-item class="more-operations" label="对比度">
            <el-slider
              v-model="state.contrast.value"
              :max="100"
              :min="-100"
              @change="
                (value) => handleImageFilterChange(Number(value), 'contrast')
              "
            />
          </el-form-item>
        </template>

        <template v-if="state.saturate.enabled && !disableMore">
          <el-form-item class="more-operations" label="饱和度">
            <el-slider
              v-model="state.saturate.value"
              :max="100"
              :min="-100"
              @change="
                (value) => handleImageFilterChange(Number(value), 'saturate')
              "
            />
          </el-form-item>
        </template>
      </el-form>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.sidebar-card {
  width: 35vw;
  height: 100%;
  padding: 0 15px;
  overflow-y: auto;
  text-align: center;
}

@media screen and (width <= 992px) {
  .sidebar-card {
    width: 96vw;
  }
}

@media screen and (width <= 767px) {
  .sidebar-card {
    width: 94vw;
  }
}
</style>

<style lang="scss">
.more-operations {
  margin-bottom: 5px;
}
</style>
