<script lang="ts" setup>
import { loadImage } from "@/utils";
import { PropType } from "vue";
import { transform } from "../AlgorithmToolBar/utils";

defineOptions({
  name: "Magnifier",
});

const props = defineProps({
  shape: {
    type: String,
    validator: (val: string) => {
      return ["circle", "square"].includes(val);
    },
    default: "square",
  },
  radius: {
    type: Number,
    default: 100,
  },
  scale: {
    type: Number,
    default: 5,
  },
  originScale: {
    type: Number,
    required: true,
  },
  src: {
    type: String,
    required: true,
  },
  bigImgSrc: {
    type: String,
    required: false,
  },
  brightness: {
    type: Number,
    default: 0,
  },
  contrast: {
    type: Number,
    default: 0,
  },
  point: {
    type: Object as PropType<{ x: number; y: number }>,
    required: true,
  },
  label: {
    type: Object as PropType<{
      text: string;
      color: string;
      backgroundColor: string;
    }>,
    required: false,
  },
});

const canvasRef = ref<HTMLCanvasElement>();
const ctxRef = ref<CanvasRenderingContext2D>();
const img = ref<HTMLImageElement>();
const trueScale = ref(1);
const width = ref(0);
const height = ref(0);
const swidth = ref(0);
const sheight = ref(0);

function initCanvas() {
  const { radius, shape } = props;
  width.value = radius * 2;
  height.value = width.value;

  const canvas = canvasRef.value!;
  canvas.width = width.value;
  canvas.height = height.value;

  if (shape === "circle") {
    canvas.style.borderRadius = "50%";
  } else {
    canvas.style.borderRadius = "0";
  }
}

async function initImage() {
  const { src, bigImgSrc, scale, originScale } = props;
  let originImg = await loadImage(src, true);
  swidth.value = originImg.width / scale;
  sheight.value = swidth.value;

  if (bigImgSrc) {
    let bigImg = await loadImage(bigImgSrc, true);
    let enlargeScale = bigImg.width / originImg.width;
    img.value = bigImg;
    swidth.value = swidth.value * enlargeScale;
    sheight.value = sheight.value * enlargeScale;
    trueScale.value = enlargeScale * originScale;
  } else {
    trueScale.value = originScale;
    img.value = originImg;
  }
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
    const { point } = props;
    let sx = Math.max(
      0,
      Math.min(
        point.x / trueScale.value - sheight.value / 2,
        img.value!.width - sheight.value
      )
    );
    let sy = Math.max(
      0,
      Math.min(
        point.y / trueScale.value - swidth.value / 2,
        img.value!.height - swidth.value
      )
    );

    ctx.clearRect(0, 0, width.value, height.value);
    ctx.drawImage(
      img.value,
      sx,
      sy,
      swidth.value,
      sheight.value,
      0,
      0,
      width.value,
      height.value
    );

    drawLabel();
  }
}

watch([() => props.radius, () => props.shape], async () => {
  initCanvas();
  await initImage();
  drawImageOnMouseMove();
});

watch(
  [
    () => props.scale,
    () => props.originScale,
    () => props.src,
    () => props.bigImgSrc,
  ],
  async () => {
    await initImage();
    drawImageOnMouseMove();
  }
);

watch(
  () => props.point,
  () => drawImageOnMouseMove(),
  { deep: true }
);

watch([() => props.brightness, () => props.contrast], () => {
  if (ctxRef.value) {
    ctxRef.value.filter = `brightness(${transform(props.brightness)}%) contrast(${transform(props.contrast)}%)`;
    drawImageOnMouseMove();
  }
});

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
