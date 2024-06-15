<script lang="ts" setup>
import { loadImage } from "@/utils";
import { PropType } from "vue";

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
  point: {
    type: Object as PropType<{ x: number; y: number }>,
    required: true,
  },
});

const canvasRef = ref<HTMLCanvasElement>();
const ctx = ref<CanvasRenderingContext2D>();
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
  sheight.value = originImg.height;
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

function drawImageOnMouseMove() {
  if (ctx.value && img.value) {
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

    ctx.value.clearRect(0, 0, width.value, height.value);
    ctx.value.drawImage(
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

onMounted(async () => {
  ctx.value = canvasRef.value!.getContext("2d")!;
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
