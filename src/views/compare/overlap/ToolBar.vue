<script lang="ts" setup>
import Magnifier from "@/components/Magnifier/index.vue";
import { PropType } from "vue";

const props = defineProps({
  imgUrls: {
    type: Array<string>,
    required: true,
  },
  point: {
    type: Object as PropType<{ x: number; y: number }>,
    required: true,
  },
  originScale: {
    type: Number,
    required: true,
  },
});

const emit = defineEmits([
  "update:contrast",
  "update:brightness",
  "toggle:magnifier",
  "toggle:takePhoto",
  "upload:image",
  "generate:image",
  "reset",
]);

const { width, height } = useWindowSize();

const magnifierInfo = reactive({
  enabled: false,
  shape: "square",
  radius: 110,
  scale: 8,
});

const contrast = ref(0);
const brightness = ref(0);

const showContrast = ref(false);
const showBrightness = ref(false);

function handleShowMagnifier() {
  magnifierInfo.enabled = !magnifierInfo.enabled;
  emit("toggle:magnifier", magnifierInfo.enabled);
}

function handleShowContrast() {
  showContrast.value = !showContrast.value;
}

function handleShowBrightness() {
  showBrightness.value = !showBrightness.value;
}

function handleTakePhoto() {
  emit("toggle:takePhoto");
}

function handleUploadChange(file: any) {}
function handleUploadExceed() {
  ElMessage.warning("最多只能上传一张图片");
}
onMounted(() => {
  magnifierInfo.radius = Math.floor((width.value * 0.35 - 90) / 4);
  watch(
    () => width.value,
    (newWidth) => {
      magnifierInfo.radius = Math.floor((newWidth * 0.35 - 90) / 4);
    }
  );
});
</script>

<template>
  <el-card>
    <h2>图像去雾</h2>
    <el-text
      >通过对遭受雾气影响的图像进行相应处理，恢复图像原本的纹理结构和细节信息，进而提升图像的能见度</el-text
    >
    <div>
      <el-upload
        :show-file-list="false"
        :limit="1"
        action="#"
        :auto-upload="false"
        :on-change="handleUploadChange"
        :on-exceed="handleUploadExceed"
        accept="image/gif,image/png,image/jpg,image/jpeg"
        >本地上传</el-upload
      >
      <el-button @click="emit('reset')">清除结果</el-button>
    </div>
    <template v-if="magnifierInfo.enabled">
      <Magnifier
        v-for="imgUrl in props.imgUrls"
        :key="imgUrl"
        :point="props.point"
        :src="imgUrl"
        :shape="magnifierInfo.shape"
        :radius="magnifierInfo.radius"
        :scale="magnifierInfo.scale"
        :origin-scale="originScale"
      />
    </template>
  </el-card>
</template>

<style lang="scss" scoped></style>
