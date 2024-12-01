<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import { CSSProperties } from "vue";

const imageShowStore = useImageShowStore();
const { mask, imageInfo, magnifierInfo, scaleX, scaleY, mouse } =
  toRefs(imageShowStore);
const { images, brightness, contrast, saturate } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);
const { enabled: isMagnifierEnabled } = toRefs(magnifierInfo.value);

const loadedCount = ref(0);

const containerStyle = ref<CSSProperties>({
  flexDirection: "row",
});

const wrapperStyle = ref<CSSProperties>({
  width: 0,
  height: 0,
});

const maskStyle = computed<CSSProperties>(() => {
  return {
    left: mask.value.x + "px",
    top: mask.value.y + "px",
    width: maskWidth.value + "px",
    height: maskHeight.value + "px",
    display: isMagnifierEnabled.value ? "block" : "none",
  };
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

function adjustSizes() {
  const length = imgUrls.value.length;
  if (!containerRef.value || length === 0) return;
  const container = containerRef.value;

  const containerWidth = container.offsetWidth;
  const containerHeight = container.offsetHeight;
  const containerWidthAspectRatio = containerWidth / length / containerHeight;

  const img = new Image();
  img.src = imgUrls.value[0].url;
  img.onload = function () {
    const imgAspectRatio = img.naturalWidth / img.naturalHeight;
    let width: number;
    let height: number;
    if (containerWidth < containerHeight) {
      containerStyle.value.flexDirection = "column";
      if (imgAspectRatio > containerWidthAspectRatio) {
        width = (containerHeight / length) * imgAspectRatio;
        height = containerHeight / length;
      } else {
        width = containerHeight;
        height = containerHeight / imgAspectRatio;
      }
    } else {
      containerStyle.value.flexDirection = "row";
      if (imgAspectRatio > containerWidthAspectRatio) {
        width = containerWidth / length;
        height = containerWidth / length / imgAspectRatio;
      } else {
        width = containerHeight / length;
        height = containerHeight / length / imgAspectRatio;
      }
    }
    wrapperStyle.value.width = `${width}px`;
    wrapperStyle.value.height = `${height}px`;

    imageShowStore.setImageSize(width, height);
    imageShowStore.setImageNaturalSize(img.naturalWidth, img.naturalHeight);
  };
}

function handleMaskMove(e: MouseEvent | TouchEvent, index: number) {
  if (wrapperRefs.value.length === 0) return;
  e.preventDefault(); // 禁用默认行为

  const containerRect = wrapperRefs.value[index].getBoundingClientRect();

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

const isMousedown = ref(false);

function mousedown(e: MouseEvent | TouchEvent, key: number) {
  isMousedown.value = true;
  magnifierStyle.value.display = "block";
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
  imageShowStore.setMouseXY(clientX, clientY);
  handleMouseEvent(e, key);
}

function mouseup(e: MouseEvent | TouchEvent, key: number) {
  isMousedown.value = false;
  magnifierStyle.value.display = "none";
}

function mousemove(e: MouseEvent | TouchEvent, key: number) {
  if (isMousedown.value) {
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
    imageShowStore.setMouseXY(clientX, clientY);
    handleMouseEvent(e, key);
  }
}

function mousewheel(e: WheelEvent, key: number) {
  if (isMousedown.value) {
    let zoomLevel = magnifierInfo.value.zoomLevel;
    zoomLevel += e.deltaY > 0 ? -0.1 : 0.1;
    zoomLevel = Math.min(Math.max(zoomLevel, 1), 10); // 保持放大倍率在1到10之间
    imageShowStore.setMagnifierZoomLevel(zoomLevel);
    console.log(e.clientX, mouse.value.x, key);
    const event = { clientX: mouse.value.x, clientY: mouse.value.y } as
      | MouseEvent
      | TouchEvent;
    handleMouseEvent(event, key);
  }
}

const maskWidth = computed(
  () => magnifierInfo.value.width / magnifierInfo.value.zoomLevel
);
const maskHeight = computed(
  () => magnifierInfo.value.height / magnifierInfo.value.zoomLevel
);

function handleMouseEvent(e: MouseEvent | TouchEvent, key: number) {
  if (
    wrapperRefs.value.length === 0 ||
    magnifierRefs.value.length === 0 ||
    imgRefs.value.length === 0
  ) {
    return;
  }

  const containerRect = wrapperRefs.value[key].getBoundingClientRect();

  // 判断事件来源是触摸还是鼠标
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
  console.log("handle", clientX, key);
  const relativeX = clientX - containerRect.left;
  const relativeY = clientY - containerRect.top;

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
      x - magnifierInfo.value.width / 4,
      imageInfo.value.width - magnifierInfo.value.width
    )
  ); // 限制放大镜在容器范围内
  const magnifierTop = Math.max(
    0,
    Math.min(
      y - magnifierInfo.value.height / 4,
      imageInfo.value.height - magnifierInfo.value.height
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

watch([maskWidth, maskHeight, scaleX, scaleY], () => {
  for (let i = 0; i < imgRefs.value.length; i++) {
    handleMouseEvent(new MouseEvent("mousemove"), i);
  }
});

onMounted(() => {
  adjustSizes();
});
</script>

<template>
  <div ref="containerRef" :style="{ ...containerStyle }" class="container">
    <div
      v-for="urls in imgUrls"
      :key="urls.id"
      :ref="setWrapperRef"
      :style="{ ...wrapperStyle }"
      class="image-wrapper"
      @mouseup="(e) => mouseup(e, urls.id)"
      @touchend="(e) => mouseup(e, urls.id)"
      @mousedown.prevent="(e) => mousedown(e, urls.id)"
      @mousemove.prevent="(e) => mousemove(e, urls.id)"
      @touchmove.prevent="(e) => mousemove(e, urls.id)"
      @touchstart.prevent="(e) => mousedown(e, urls.id)"
      @wheel.prevent="(e) => mousewheel(e, urls.id)"
    >
      <img
        :ref="setImgRef"
        :src="urls.url"
        :style="{
          ...wrapperStyle,
          filter: `contrast(${contrast}%) brightness(${brightness}%) saturate(${saturate}%)`,
        }"
        alt=""
        @load="imageOnload"
      />
      <!--      <div :style="maskStyle" class="mask"></div>-->
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
.container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 80vw;
  height: 90vh;
  overflow: hidden;
  border: 2px solid #ccc;
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

.mask {
  position: absolute;
  z-index: 5;
  display: none; /* 初始隐藏 */
  width: 100px;
  height: 100px;
  pointer-events: none;
  background-color: rgb(255 255 255 / 20%);
  border: 2px solid rgb(255 255 255 / 80%);
}

.magnifier {
  position: absolute;
  z-index: 5;
  display: none; /* 初始隐藏 */
  pointer-events: none;
  border: 2px solid rgb(255 255 255 / 80%);
}
</style>
