<script lang="ts" setup>
import { useImageShowStore } from "@/store/modules/imageShow";
import { hexToRGBA } from "@/utils";
import { CSSProperties } from "vue";
import { ModelAPI, CompareReportForm } from "dehaze-sdk-js";
import { Download } from "@element-plus/icons-vue";

const imageShowStore = useImageShowStore();

defineOptions({ name: "CompareParallel" });
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

function imageOnerror() {
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
    const magnifierCtx = magnifier.getContext("2d");
    if (!magnifierCtx) return;
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

// Export report
const reportDialogVisible = ref(false);
const reportGenerating = ref(false);
const reportForm = ref<CompareReportForm>({
  logId: 0,
  format: "pdf",
  includeMetrics: true,
  includeFilters: false,
});

function openReportDialog() {
  reportForm.value.logId = 0;
  reportDialogVisible.value = true;
}

async function handleExportReport() {
  if (reportForm.value.logId === 0) {
    ElMessage.warning("当前没有可导出的对比记录");
    return;
  }
  reportGenerating.value = true;
  try {
    const res = await ModelAPI.generateReport(reportForm.value);
    if (!res.taskId) {
      throw new Error("未返回任务ID");
    }
    while (true) {
      const status = await ModelAPI.getReportStatus(res.taskId);
      if (status.status === 2) {
        if (status.downloadUrl) {
          const link = document.createElement("a");
          link.href = status.downloadUrl;
          link.download = `dehaze-report.${reportForm.value.format}`;
          link.click();
        } else {
          ElMessage.success("报告生成完成，请前往任务中心下载");
        }
        break;
      }
      if (status.status === 3) {
        throw new Error(status.errorMessage || "报告生成失败");
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    reportDialogVisible.value = false;
  } catch (e: any) {
    ElMessage.error("导出报告失败：" + (e.message || "未知错误"));
  } finally {
    reportGenerating.value = false;
  }
}

onMounted(() => {
  adjustSizes();
});
</script>

<template>
  <div class="parallel-header">
    <el-button
      type="primary"
      @click="openReportDialog"
      :loading="reportGenerating"
    >
      <el-icon><Download /></el-icon>
      导出对比报告
    </el-button>
  </div>
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
        @error="imageOnerror"
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

    <el-dialog v-model="reportDialogVisible" title="导出对比报告" width="420px">
      <el-form label-position="top">
        <el-form-item label="报告格式">
          <el-radio-group v-model="reportForm.format">
            <el-radio label="pdf">PDF</el-radio>
            <el-radio label="image">图片 (PNG)</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="包含内容">
          <el-checkbox v-model="reportForm.includeMetrics"
            >包含评价指标</el-checkbox
          >
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="reportDialogVisible = false">取消</el-button>
        <el-button
          type="primary"
          @click="handleExportReport"
          :loading="reportGenerating"
        >
          导出
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.parallel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: #fff;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

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
