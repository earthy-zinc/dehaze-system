<script lang="ts" setup>
import DraggableLine from "@/components/DraggableLine/index.vue";
import { transform } from "@/components/AlgorithmToolBar/utils";
import { CSSProperties } from "vue";

defineOptions({
  name: "OverlapImageShow",
});

const props = defineProps({
  image1: {
    type: String,
    required: true,
  },
  image2: {
    type: String,
    required: true,
  },
  showMask: {
    type: Boolean,
    default: false,
  },
  contrast: {
    type: Number,
    default: 0,
  },
  brightness: {
    type: Number,
    default: 0,
  },
});

const emit = defineEmits(["onOriginScaleChange", "onMouseover"]);

const { image1, image2, showMask, contrast, brightness } = toRefs(props);

const maskRef = ref<HTMLDivElement>();
const maskStyle = ref<CSSProperties>({
  width: "100px",
  height: "100px",
  backgroundColor: "rgba(0, 0, 0, 0.5)",
  position: "absolute",
  display: "none",
});

const image1Ref = ref<HTMLImageElement>();
const { elementX, elementY, isOutside } = useMouseInElement(image1Ref);

const { height: imgHeight, width: imgWidth } = useElementSize(image1Ref);

const originScale = computed(() => {
  if (image1Ref.value) return imgHeight.value / image1Ref.value.height;
  return 1;
});

const sliderPosition = ref(0);
const sliderValue = computed(() => {
  return (1 - sliderPosition.value / imgWidth.value) * 100;
});

function updateImageStyle() {
  const images = document.querySelectorAll(".image-container img");
  images.forEach((img: Element) => {
    (img as HTMLImageElement).style.filter = `
    contrast(${transform(contrast.value)}%)
    brightness(${transform(brightness.value)}%)`;
  });
}

function mousemove(event: MouseEvent) {
  if (isOutside.value) return;

  function getMaskStyleTopAndLeft() {
    let top = elementY.value - maskRef.value!.offsetHeight / 2;
    let left = elementX.value - maskRef.value!.offsetWidth / 2;

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

onMounted(() => {
  watch([contrast, brightness], () => updateImageStyle());
  watch([originScale], () => emit("onOriginScaleChange", originScale.value));

  watch([isOutside, elementX, elementY], () => {
    if (!isOutside.value)
      emit("onMouseover", { x: elementX.value, y: elementY.value });
  });
});
</script>

<template>
  <div>
    <div
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
  </div>
</template>

<style lang="scss" scoped>
.image-container {
  position: relative;
  width: 100%; /* 容器宽度与父容器相同 */
  height: 100%; /* 容器高度与父容器相同 */
  overflow: hidden;
}

.image-container img {
  position: absolute;
  width: 100%; /* 让图片宽度填充容器 */
  height: 100%; /* 让图片高度填充容器 */
  object-fit: contain; /* 保持图片宽高比，缩放以适应容器 */
}

.mouse-mask {
  position: fixed;
  cursor: crosshair;
  background-color: rgb(0 0 0 / 50%);
}
</style>
