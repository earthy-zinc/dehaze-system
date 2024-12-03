<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import { CSSProperties } from "vue";
import { hexToRGBA } from "@/utils";

defineOptions({ name: "ParallelImageShow" });

const imageShowStore = useImageShowStore();
const { imageInfo, magnifierInfo, scaleX, scaleY, mouse } =
  toRefs(imageShowStore);
const { images, brightness, contrast, saturate } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);
const loadedCount = ref(0);

const containerStyle = ref<CSSProperties>({
  flexDirection: "row",
});

const wrapperStyle = ref<CSSProperties>({
  width: 0,
  height: 0,
});

const magnifierStyle = ref<CSSProperties>({
  left: 0,
  top: 0,
  display: "none",
});

const containerRef = ref<HTMLDivElement>();
const imgRefs = ref<HTMLImageElement[]>([]);
const wrapperRefs = ref<HTMLDivElement[]>([]);
const magnifierRefs = ref<HTMLCanvasElement[]>([]);

function setWrapperRef(ref: Element | ComponentPublicInstance | null) {
  if (ref instanceof HTMLDivElement && !wrapperRefs.value.includes(ref)) {
    wrapperRefs.value.push(ref);
  }
}

function setMagnifierRef(ref: Element | ComponentPublicInstance | null) {
  if (ref instanceof HTMLCanvasElement && !magnifierRefs.value.includes(ref)) {
    magnifierRefs.value.push(ref);
  }
}

function setImgRef(ref: Element | ComponentPublicInstance | null) {
  if (ref instanceof HTMLImageElement && !imgRefs.value.includes(ref)) {
    imgRefs.value.push(ref);
  }
}

function imageOnload() {
  loadedCount.value++;
  if (loadedCount.value === imgUrls.value.length) {
    adjustSizes();
  }
}

const { width: windowWidth, height: windowHeight } = useWindowSize();

function adjustSizes() {
  const length = imgUrls.value.length;
  if (length === 0) return;
  const containerWidth = windowWidth.value - 80;
  const containerHeight = windowHeight.value;
  const containerWidthAspectRatio = containerWidth / length / containerHeight;

  const img = new Image();
  img.src = imgUrls.value[0].url;
  img.onload = function () {
    const imgAspectRatio = img.naturalWidth / img.naturalHeight;
    let width: number;
    let height: number;
    if (containerWidth < containerHeight) {
      if (imgAspectRatio > containerWidthAspectRatio) {
        width = (containerHeight / length) * imgAspectRatio;
        height = containerHeight / length;
      } else {
        width = containerHeight;
        height = containerHeight / imgAspectRatio;
      }
      containerStyle.value.flexDirection = "column";
      containerStyle.value.height = `${height * length}px`;
    } else {
      if (imgAspectRatio > containerWidthAspectRatio) {
        width = containerWidth / length;
        height = containerWidth / length / imgAspectRatio;
      } else {
        width = containerHeight / length;
        height = containerHeight / length / imgAspectRatio;
      }
      containerStyle.value.flexDirection = "row";
      containerStyle.value.height = `${height}px`;
    }
    wrapperStyle.value.width = `${width}px`;
    wrapperStyle.value.height = `${height}px`;

    imageShowStore.setImageSize(width, height);
    imageShowStore.setImageNaturalSize(img.naturalWidth, img.naturalHeight);
  };
}

const isMousedown = ref(false);
const selectedWrapperRect = reactive({
  left: 0,
  top: 0,
});

function mousedown(e: MouseEvent | TouchEvent) {
  isMousedown.value = true;
  magnifierStyle.value.display = "block";
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
  const containerRect = (e.target as HTMLElement).getBoundingClientRect();
  selectedWrapperRect.left = containerRect.left;
  selectedWrapperRect.top = containerRect.top;
  imageShowStore.setMouseXY(clientX, clientY);
  handleMouseEvent();
}

function mouseup(e: MouseEvent | TouchEvent) {
  isMousedown.value = false;
  magnifierStyle.value.display = "none";
}

function mousemove(e: MouseEvent | TouchEvent) {
  if (isMousedown.value) {
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
    const containerRect = (e.target as HTMLElement).getBoundingClientRect();
    selectedWrapperRect.left = containerRect.left;
    selectedWrapperRect.top = containerRect.top;
    imageShowStore.setMouseXY(clientX, clientY);
    handleMouseEvent();
  }
}

const zoomIn = ref(true);

function mousewheel(e: WheelEvent) {
  if (isMousedown.value) {
    let zoomLevel = magnifierInfo.value.zoomLevel;
    zoomIn.value = e.deltaY < 0;
    zoomLevel += e.deltaY > 0 ? -0.2 : 0.2;
    zoomLevel = Math.min(Math.max(zoomLevel, 1), 10); // 保持放大倍率在1到10之间
    imageShowStore.setMagnifierZoomLevel(zoomLevel);
    handleMouseEvent();
  }
}

const maskWidth = computed(
  () => magnifierInfo.value.width / magnifierInfo.value.zoomLevel
);
const maskHeight = computed(
  () => magnifierInfo.value.height / magnifierInfo.value.zoomLevel
);

function handleMouseEvent() {
  if (
    wrapperRefs.value.length === 0 ||
    magnifierRefs.value.length === 0 ||
    imgRefs.value.length === 0
  ) {
    return;
  }

  const relativeX = mouse.value.x - selectedWrapperRect.left;
  const relativeY = mouse.value.y - selectedWrapperRect.top;

  const x = Math.max(
    0,
    Math.min(
      relativeX - maskWidth.value / 2,
      imageInfo.value.width - maskWidth.value
    )
  );
  const y = Math.max(
    0,
    Math.min(
      relativeY - maskHeight.value / 2,
      imageInfo.value.height - maskHeight.value
    )
  );
  const magnifierLeft = Math.max(
    0,
    Math.min(
      relativeX - magnifierInfo.value.width / 2,
      imageInfo.value.width - magnifierInfo.value.width - 4
    )
  );
  const magnifierTop = Math.max(
    0,
    Math.min(
      relativeY - magnifierInfo.value.height / 2,
      imageInfo.value.height - magnifierInfo.value.height - 4
    )
  );
  magnifierStyle.value.left = `${magnifierLeft}px`;
  magnifierStyle.value.top = `${magnifierTop}px`;
  updateMagnifier(x, y);
}

function updateMagnifier(x: number, y: number) {
  magnifierRefs.value.forEach((magnifier, index) => {
    const magnifierCtx = magnifier.getContext("2d")!;
    magnifierCtx.clearRect(
      0,
      0,
      magnifierInfo.value.width,
      magnifierInfo.value.height
    );
    magnifierCtx.drawImage(
      imgRefs.value[index],
      x * scaleX.value,
      y * scaleY.value,
      maskWidth.value * scaleX.value,
      maskHeight.value * scaleY.value,
      0,
      0,
      magnifierInfo.value.width,
      magnifierInfo.value.height
    );
  });
}

const { width, height } = useWindowSize();
watch([width, height], () => adjustSizes());

onMounted(() => {
  adjustSizes();
});
</script>

<template>
  <div
    ref="containerRef"
    :style="{ ...containerStyle }"
    class="parallel-container"
    @mouseup="mouseup"
    @touchend="mouseup"
    @mousedown.prevent="mousedown"
    @mousemove.prevent="mousemove"
    @touchmove.prevent="mousemove"
    @touchstart.prevent="mousedown"
    @wheel.prevent="mousewheel"
  >
    <div
      v-for="urls in imgUrls"
      :key="urls.id"
      :ref="setWrapperRef"
      :style="{ ...wrapperStyle }"
      class="image-wrapper"
    >
      <img
        :ref="setImgRef"
        :src="urls.url"
        :style="{
          ...wrapperStyle,
          filter: `contrast(${contrast}%) brightness(${brightness}%) saturate(${saturate}%)`,
          cursor: zoomIn ? 'zoom-in' : 'zoom-out',
        }"
        alt=""
        @load="imageOnload"
      />
      <div
        :style="{
          backgroundColor: hexToRGBA(urls.label.backgroundColor, 0.5),
          color: urls.label.color,
        }"
        class="label left-label"
      >
        <span>{{ urls.label.text }}</span>
      </div>
      <canvas
        :ref="setMagnifierRef"
        :height="magnifierInfo.height"
        :style="magnifierStyle"
        :width="magnifierInfo.width"
        class="magnifier"
      ></canvas>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.parallel-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.image-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.image-wrapper img {
  width: 100%;
  height: 100%;
  object-fit: contain; /* 保证图片宽高比 */
}

.magnifier {
  position: absolute;
  z-index: 5;
  display: none; /* 初始隐藏 */
  pointer-events: none;
  border: 2px solid rgb(255 255 255 / 80%);
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
  left: 0;
  background-color: rgb(162 162 162 / 50%);
}
</style>
