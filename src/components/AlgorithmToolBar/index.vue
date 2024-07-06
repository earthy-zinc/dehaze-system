<script lang="ts" setup>
import { MagnifierInfo } from "./types";
import AlgorithmHeader from "@/components/AlgorithmToolBar/AlgorithmHeader/index.vue";
import ImageUploadButton from "@/components/AlgorithmToolBar/ImageUploadButton/index.vue";
import SubmitButton from "@/components/AlgorithmToolBar/SubmitButton/index.vue";

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
  magnifier: {
    type: Object as () => MagnifierInfo,
    default: () => ({
      imgUrls: [],
      radius: 100,
      originScale: 1,
      point: { x: 0, y: 0 },
    }),
  },
});

const emit = defineEmits([
  "onUpload",
  "onTakePhoto",
  "onReset",
  "onGenerate",
  "onMagnifierChange",
  "onBrightnessChange",
  "onContrastChange",
]);
</script>

<template>
  <div class="mr-3">
    <el-card class="sidebar-card">
      <AlgorithmHeader :description="description" :title="title" />
      <ImageUploadButton
        @on-upload="(file) => emit('onUpload', file)"
        @on-take-photo="() => emit('onTakePhoto')"
        @on-reset="() => emit('onReset')"
      />
      <slot></slot>
      <SubmitButton
        :disable-more="disableMore"
        :magnifier="magnifier"
        @on-generate="() => emit('onGenerate')"
        @on-magnifier-change="(flag) => emit('onMagnifierChange', flag)"
        @on-brightness-change="(value) => emit('onBrightnessChange', value)"
        @on-contrast-change="(value) => emit('onContrastChange', value)"
      />
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.sidebar-card {
  width: 35vw;
  height: 100%;
  padding: 0 15px;
  overflow-y: auto;
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
