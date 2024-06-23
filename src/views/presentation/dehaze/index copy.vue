<script lang="ts" setup>
import { FileInfo } from "@/api/file/model";
import DraggableLine from "@/components/DraggableLine/index.vue";
import ToolBar from "@/views/presentation/dehaze/ToolBar.vue";
import { CSSProperties } from "vue";

const { width, height } = useWindowSize();
const image1 = ref("https://picsum.photos/id/1/800/800");
const image2 = ref("https://picsum.photos/id/2/800/800");
const imageContainerRef = ref();
const image1Ref = ref<HTMLImageElement>();
const maskRef = ref<HTMLDivElement>();

const contrast = ref(0);
const brightness = ref(0);
const sliderPosition = ref(0);
const imgWidth = ref(0);
const imgHeight = ref(0);
const showMask = ref(false);
const maskStyle = ref<CSSProperties>({
  width: "100px",
  height: "100px",
  backgroundColor: "rgba(0, 0, 0, 0.5)",
  position: "absolute",
  display: "none",
});

const point = ref({ x: 0, y: 0 });
const originScale = ref(1);
const cardRef = ref<HTMLElement>();
const toolBarWidth = ref(0);

watch(
  () => image1Ref.value?.height,
  () => {
    if (image1Ref.value) {
      originScale.value =
        image1Ref.value.height / image1Ref.value.naturalHeight;
    }
  }
);

onMounted(() => {
  image1Ref.value!.onload = () => {
    imgWidth.value = image1Ref.value!.width;
    imgHeight.value = image1Ref.value!.height;
    originScale.value =
      image1Ref.value!.height / image1Ref.value!.naturalHeight;
    toolBarWidth.value = width.value - 100 - cardRef.value!.clientWidth;
    imageContainerRef.value.style.width = `${image1Ref.value!.width}px`;
  };
});

const sliderValue = computed(() => {
  return (1 - sliderPosition.value / imgHeight.value) * 100;
});

function mousemove(event: MouseEvent) {
  const { left, top } = image1Ref.value!.getBoundingClientRect();
  point.value.x = event.clientX - left;
  point.value.y = event.clientY - top;
  console.log(point.value, left, top, event.offsetX, event.clientY);

  function getMaskStyleTopAndLeft() {
    let top = point.value.y - maskRef.value!.offsetHeight / 2;
    let left = point.value.x - maskRef.value!.offsetWidth / 2;

    if (top + maskRef.value!.offsetHeight > imgHeight.value) {
      top = imgHeight.value - maskRef.value!.offsetHeight;
    }
    if (left + maskRef.value!.offsetWidth > imgWidth.value) {
      left = imgWidth.value - maskRef.value!.offsetWidth;
    }
    if (top < 0) {
      top = 0;
    }
    if (left < 0) {
      left = 0;
    }
    return { maskTop: top, maskLeft: left };
  }

  let { maskTop, maskLeft } = getMaskStyleTopAndLeft();
  maskStyle.value.top = maskTop + "px";
  maskStyle.value.left = maskLeft + "px";
}

function mouseover() {
  maskStyle.value.display = "block";
}

function mouseleave() {
  maskStyle.value.display = "none";
}

const handleContrastChange = (value: number) => {
  contrast.value = value;
  updateImageStyle();
};

const handleBrightnessChange = (value: number) => {
  brightness.value = value;
  updateImageStyle();
};

function updateImageStyle() {
  function transform(x: number) {
    return 0.5 * x + 100;
  }

  const images = document.querySelectorAll(".image-container img");
  images.forEach((img: Element) => {
    (img as HTMLImageElement).style.filter = `
    contrast(${transform(contrast.value)}%)
    brightness(${transform(brightness.value)}%)`;
  });
}

function handleShowMagnifier(showMagnifier: boolean) {
  showMask.value = showMagnifier;
}

function handleUploadFile(fileInfo: FileInfo) {
  image1.value = fileInfo.url;
  image2.value = "";
}

function handleGenerateImage() {}

function handleReset() {
  image1.value = "";
  image2.value = "";
  contrast.value = 0;
  brightness.value = 0;
}
</script>

<template>
  <div class="app-container">
    <ToolBar
      :img-urls="[image1, image2]"
      :origin-scale="originScale"
      :point="point"
      :style="{ width: toolBarWidth ? toolBarWidth + 'px' : 'auto' }"
      @reset="handleReset"
      @update:brightness="handleBrightnessChange"
      @update:contrast="handleContrastChange"
      @upload:image="handleUploadFile"
      @generate:image="handleGenerateImage"
      @toggle:magnifier="handleShowMagnifier"
    />

    <el-card ref="cardRef" class="flex-center" style="width: 64vw">
      <div v-if="image1 === '' && image2 === ''">试试样张吧</div>
      <div
        v-else
        ref="imageContainerRef"
        :style="{ width: image1Ref ? image1Ref.width + 'px' : 'auto' }"
        class="image-container"
        @mouseleave="mouseleave"
        @mousemove="mousemove"
        @mouseover="mouseover"
      >
        <img
          ref="image1Ref"
          :src="image1"
          :style="{ clipPath: `inset(0 ${sliderValue}% 0 0)` }"
          alt=""
        />
        <img
          :src="image2"
          :style="{ clipPath: `inset(0 0 0 ${100 - sliderValue}%)` }"
          alt=""
        />
        <div
          v-show="showMask"
          ref="maskRef"
          :style="maskStyle"
          class="mouse-mask"
        ></div>
        <DraggableLine
          left-label="原图"
          right-label="对比图"
          @update:offset="(value) => (sliderPosition = value)"
        />
      </div>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.app-container {
  display: flex;
}

.image-container {
  position: relative;
  height: calc(100vh - $navbar-height - 22px - 2 * var(--el-card-padding));
  overflow: hidden;
}

.image-container img {
  position: absolute;
  width: auto;
  height: 100%;
  object-fit: cover;
}

.mouse-mask {
  position: fixed;
  cursor: crosshair;
  background-color: rgb(0 0 0 / 50%);
}
</style>
