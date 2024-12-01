<script lang="ts" setup>
import { LabelType, useImageShowStore } from "@/store/modules/imageShow";
import { PropType } from "vue";
import { loadImage } from "@/utils";

const props = defineProps({
  src: {
    type: String,
    required: true,
  },
  label: {
    type: Object as PropType<LabelType>,
    required: false,
  },
});

const canvasRef = ref<HTMLCanvasElement>();
const ctxRef = ref<CanvasRenderingContext2D>();
const img = ref<HTMLImageElement>();

const overlapImageStore = useImageShowStore();
const {
  scaleX,
  scaleY,
  mask,
  maskWidth,
  maskHeight,
  magnifierInfo,
  imageInfo,
} = toRefs(overlapImageStore);

const { width, height, shape } = toRefs(magnifierInfo.value);
const { brightness, contrast, saturate } = toRefs(imageInfo.value);

function initCanvas() {
  const canvas = canvasRef.value!;
  canvas.width = width.value;
  canvas.height = height.value;
  if (shape.value === "circle") {
    canvas.style.borderRadius = "50%";
  } else {
    canvas.style.borderRadius = "0";
  }
}

async function initImage() {
  img.value = await loadImage(props.src, true);
}

function drawLabel() {
  if (!props.label) return;
  const { text, color, backgroundColor } = props.label;
  const ctx = ctxRef.value!;
  ctx.font = "15px sans-serif";
  ctx.fillStyle = backgroundColor;
  ctx.globalAlpha = 0.2;

  let metrics = ctx.measureText(text);
  let textWidth = metrics.width + 10;
  ctx.fillRect(0, 0, textWidth, 20);

  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.fillText(text, 3, 15);
}

function drawImageOnMouseMove() {
  if (ctxRef.value && img.value) {
    const ctx = ctxRef.value;
    ctx.filter = `
      brightness(${brightness.value}%)
      contrast(${contrast.value}%)
      saturate(${saturate.value}%)
    `;
    ctx.clearRect(0, 0, width.value, height.value);
    ctx.drawImage(
      img.value,
      mask.value.x * scaleX.value,
      mask.value.y * scaleY.value,
      maskWidth.value * scaleX.value,
      maskHeight.value * scaleY.value,
      0,
      0,
      width.value,
      height.value
    );
    drawLabel();
  }
}

watch(
  [width, height, shape],
  async () => {
    initCanvas();
    drawImageOnMouseMove();
  },
  { deep: true }
);

watch(
  () => props.src,
  async () => {
    await initImage();
    drawImageOnMouseMove();
  }
);

watch(
  [mask, maskWidth, maskHeight, scaleY, scaleX, brightness, contrast, saturate],
  () => drawImageOnMouseMove(),
  { deep: true }
);

onMounted(async () => {
  ctxRef.value = canvasRef.value!.getContext("2d")!;
  initCanvas();
  await initImage();
  drawImageOnMouseMove();
});
</script>

<template>
  <canvas ref="canvasRef"></canvas>
</template>

<style lang="scss" scoped>
canvas {
  background-color: transparent;
}
</style>
