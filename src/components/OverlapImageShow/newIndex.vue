<script lang="ts" setup>
import { useSettingsStore } from "@/store";
import { useImageShowStore } from "@/store/modules/imageShow";
import { CSSProperties } from "vue";
import { hexToRGBA } from "@/utils";

const settingStore = useSettingsStore();
const imageShowStore = useImageShowStore();
const { mask, maskWidth, maskHeight, imageInfo, dividerInfo, magnifierInfo } =
  toRefs(imageShowStore);
const { images, brightness, contrast, saturate } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);
const { enabled: isMagnifierEnabled } = toRefs(magnifierInfo.value);

const loadedCount = ref(0);
const dividerPercentage = ref(50);

const containerRef = ref<HTMLDivElement>();
const containerStyle = ref<CSSProperties>({
  width: "100%",
  height: "100%",
});

const cardRef = ref<HTMLDivElement>();

const maskRef = ref<HTMLDivElement>();
const maskStyle = computed<CSSProperties>(() => {
  return {
    left: mask.value.x + "px",
    top: mask.value.y + "px",
    width: maskWidth.value + "px",
    height: maskHeight.value + "px",
    display: isMagnifierEnabled.value ? "block" : "none",
  };
});

const particleCanvasRef = ref<HTMLCanvasElement>();

function playParticleEffect() {
  if (!particleCanvasRef.value && !cardRef.value) return;
  const canvas = particleCanvasRef.value!;
  const container = cardRef.value!;
  const ctx = canvas.getContext("2d")!;

  canvas.width = container.offsetWidth;
  canvas.height = container.offsetHeight;

  const particles = [] as any[];
  const colors = ["#FF6347", "#FFD700", "#7CFC00", "#00BFFF", "#FF69B4"];

  // 修改: 添加控制持续时间和范围大小的参数
  const maxLife = 200; // 最大生命周期
  const maxSize = 8; // 最大粒子尺寸
  const minSize = 2; // 最小粒子尺寸
  const maxSpeed = 10; // 最大粒子速度
  const particleCount = 300;

  function createParticle(x: number, y: number) {
    const angle = Math.random() * Math.PI * 2;
    const speed = Math.random() * maxSpeed + 1; // 速度范围
    const size = Math.random() * (maxSize - minSize) + minSize; // 大小范围
    const color = colors[Math.floor(Math.random() * colors.length)];
    const life = Math.random() * maxLife + 50; // 生命值范围

    particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      size,
      color,
      life,
    });
  }

  function updateParticles() {
    particles.forEach((p, i) => {
      p.x += p.vx;
      p.y += p.vy;
      p.size *= 0.99; // 粒子逐渐变小
      p.life -= 1; // 生命值减少
      if (p.life <= 0 || p.size <= 0.5) {
        // 添加 size 小于某个值时也删除粒子
        particles.splice(i, 1);
      }
    });
  }

  function drawParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });
  }

  function animate() {
    updateParticles();
    drawParticles();
    if (particles.length > 0) {
      requestAnimationFrame(animate);
    }
  }

  // 播放粒子效果
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;

  // 创建粒子
  for (let i = 0; i < particleCount; i++) {
    createParticle(centerX, centerY);
  }
  animate();
}

function imageOnload() {
  loadedCount.value++;
  if (loadedCount.value === imgUrls.value.length) {
    adjustSize();
    playParticleEffect();
    dividerPercentage.value = 100;
    dividerAnimate(100, 0, 3000);
  }
}

function adjustSize() {
  if (!cardRef.value) return;
  const card = cardRef.value;
  const cardWidth = card.offsetWidth;
  const cardHeight = card.offsetHeight;
  const cardAspectRatio = cardWidth / cardHeight;

  const img = new Image();
  img.src = imgUrls.value[0].url;

  img.onload = function () {
    const imgAspectRatio = img.width / img.height;

    let width, height;
    if (imgAspectRatio > cardAspectRatio) {
      width = cardWidth;
      height = cardWidth / imgAspectRatio;
    } else {
      height = cardHeight;
      width = cardHeight * imgAspectRatio;
    }

    containerStyle.value.width = `${width}px`;
    containerStyle.value.height = `${height}px`;

    imageShowStore.setImageSize(width, height);
    imageShowStore.setImageNaturalSize(img.naturalWidth, img.naturalHeight);
  };
}

const isDraggingDivider = ref(false);

function adjustDivider(clientX: number) {
  if (!containerRef.value) return;
  const containerRect = containerRef.value.getBoundingClientRect();
  const offsetX = Math.max(
    containerRect.left,
    Math.min(clientX, containerRect.right)
  ); // 限制在容器范围内
  dividerPercentage.value =
    ((offsetX - containerRect.left) / containerRect.width) * 100;
}

function dividerAnimate(from: number, to: number, duration: number) {
  const startTime = performance.now();
  const changeInValue = to - from;
  isDraggingDivider.value = true;

  const animate = (currentTime: number) => {
    const elapsedTime = currentTime - startTime;
    const progress = Math.min(elapsedTime / duration, 1);
    dividerPercentage.value = from + changeInValue * progress;

    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      isDraggingDivider.value = false;
    }
  };
  requestAnimationFrame(animate);
}

function dividerMousedown(event: MouseEvent | TouchEvent) {
  if (dividerInfo.value.enabled) {
    isDraggingDivider.value = true;
    // 添加事件监听器
    const moveHandler = (e: MouseEvent | TouchEvent) => {
      e.preventDefault(); // 禁用默认行为
      const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
      if (isDraggingDivider.value) {
        adjustDivider(clientX);
      }
    };

    const upHandler = () => {
      isDraggingDivider.value = false;
      // 移除事件监听器
      document.removeEventListener("mousemove", moveHandler);
      document.removeEventListener("mouseup", upHandler);
      document.removeEventListener("touchmove", moveHandler);
      document.removeEventListener("touchend", upHandler);
    };

    document.addEventListener("mousemove", moveHandler);
    document.addEventListener("mouseup", upHandler);
    document.addEventListener("touchmove", moveHandler);
    document.addEventListener("touchend", upHandler);
  }
}

function handleMaskMove(e: MouseEvent | TouchEvent) {
  if (!isMagnifierEnabled.value) return;
  if (!containerRef.value) return;
  if (!maskRef.value) return;
  e.preventDefault(); // 禁用默认行为

  const containerRect = containerRef.value.getBoundingClientRect();

  // 判断事件来源是触摸还是鼠标
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;

  const x = clientX - containerRect.left - maskWidth.value / 2;
  const y = clientY - containerRect.top - maskHeight.value / 2;

  const maskX = Math.max(0, Math.min(x, containerRect.width - maskWidth.value)); // 限制mask在容器范围内
  const maskY = Math.max(
    0,
    Math.min(y, containerRect.height - maskHeight.value)
  );
  imageShowStore.setMaskXY(maskX, maskY);
}

const { width, height } = useWindowSize();

onMounted(() => {
  adjustSize();
  watch([width, height], () => adjustSize());
});
</script>

<template>
  <div ref="cardRef">
    <div
      ref="containerRef"
      :style="containerStyle"
      class="comparison-container"
      @mousemove="handleMaskMove"
      @touchmove="handleMaskMove"
    >
      <canvas ref="particleCanvasRef" class="particle-canvas"></canvas>
      <img
        :src="imgUrls[0].url"
        :style="{
          ...containerStyle,
          filter: `contrast(${contrast}%) brightness(${brightness}%) saturate(${saturate}%)`,
        }"
        alt="1"
        @load="imageOnload"
      />
      <img
        :src="imgUrls[1].url"
        :style="{
          ...containerStyle,
          filter: `contrast(${contrast}%) brightness(${brightness}%) saturate(${saturate}%)`,
          clipPath: `polygon(${dividerPercentage}% 0, 100% 0, 100% 100%, ${dividerPercentage}% 100%)`,
        }"
        alt="2"
        @load="imageOnload"
      />
      <div
        :style="{
          left: dividerPercentage + '%',
          backgroundColor: settingStore.themeColor,
          display: dividerInfo.enabled ? 'block' : 'none',
        }"
        class="divider"
        @mousedown="dividerMousedown"
        @touchstart="dividerMousedown"
      >
        <div class="label left-label">
          <span>{{ imgUrls[0].label.text }}</span>
        </div>
        <div
          :style="{ backgroundColor: hexToRGBA(settingStore.themeColor, 0.5) }"
          class="label right-label"
        >
          <span>{{ imgUrls[1].label.text }}</span>
        </div>
        <svg-icon
          :color="settingStore.themeColor"
          class="center-label"
          icon-class="hollow-slide"
          size="2em"
        />
      </div>
      <div ref="maskRef" :style="maskStyle" class="mask"></div>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.comparison-container {
  position: relative;
  overflow: hidden;
  user-select: none;

  /* 禁用全局文本选择 */
  user-select: none;
  background-color: #000;
  border: 1px solid #ccc;

  /* 禁用全局文本选择 */
}

.comparison-container img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;

  /* 保证图片宽高比 */
}

.img2 {
  z-index: 2;
  clip-path: polygon(50% 0, 100% 0, 100% 100%, 50% 100%);
}

.divider {
  position: absolute;
  top: 0;
  left: 50%;
  z-index: 3;
  width: 4px;
  height: 100%;
  cursor: ew-resize;
  background-color: #fff;
}

.mask {
  position: absolute;
  z-index: 4;
  width: 100px;
  height: 100px;
  pointer-events: none;
  background-color: rgb(0 255 0 / 10%);
  border: 2px solid rgb(0 255 0 / 70%);
}

.label {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 80px;
  height: 30px;
  line-height: 30px;
  color: var(--el-border-color);
  text-align: center;
}

.left-label {
  left: calc(50% - 82px);
  background-color: rgb(162 162 162 / 50%);
}

.right-label {
  left: 4px;
  background-color: rgb(4 27 160 / 50%);
}

.center-label {
  position: absolute;
  top: 50%;
  width: 20px;
  height: 20px;
  transform: translateX(-42%);
}

.particle-canvas {
  position: absolute;
  top: 0;
  left: 0;

  /* 不影响交互 */
  z-index: 15;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
</style>
