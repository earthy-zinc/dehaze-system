<script lang="ts" setup>
import LongitudinalWaterfall from "@/components/LongitudinalWaterfall/index.vue";
import Waterfall from "@/components/Waterfall/index.vue";
import { ViewCard } from "@/components/Waterfall/types";
import { IMAGE_TYPE_LABELS, formatHazeLevel } from "@/enums/ImageTypeEnum";
import {
  Dataset,
  DatasetAPI,
  DatasetItemAPI,
  DatasetItemQuery,
  DatasetItemVO,
  ImageUrlVO,
} from "dehaze-sdk-js";
import {
  ArrowLeft,
  ArrowRight,
  DataAnalysis,
  Delete,
  Download,
  Grid,
  List,
  Menu,
  Plus,
  Refresh,
  Search,
  Sort,
  Upload,
  UploadFilled,
} from "@element-plus/icons-vue";
import * as echarts from "echarts";
import type { UploadUserFile } from "element-plus";

defineOptions({
  name: "DataItem",
  inheritAttrs: false,
});

// ==================== 基础数据 ====================
const datasetId = ref<number>(0);
const totalPages = ref<number>(1);
const total = ref<number>(0);
const queryParams = reactive<DatasetItemQuery>({
  pageNum: 1,
  pageSize: 10,
  datasetId: 0,
});
const renderCount = ref<number>(0);
let datasetInfo = ref<Dataset>({
  id: 0,
  parentId: 0,
  name: "",
  type: "",
  description: "",
  createTime: new Date(),
  updateTime: new Date(),
  path: "",
  total: 0,
});
let imageData = reactive<DatasetItemVO[]>([]);

// 标注状态过滤：已标注/未标注二分
type AnnotationFilter = "annotated" | "unannotated";
const annotationFilter = ref<AnnotationFilter>("annotated");

let loadingBarRef = ref();
const loadingObserver = ref();

const route = useRoute();
const { width } = useWindowSize();

const itemWidth = computed(() => {
  const breakpoints = [
    { minWidth: 0, columns: 1 },
    { minWidth: 768, columns: 2 },
    { minWidth: 1024, columns: 3 },
    { minWidth: 1280, columns: 4 },
  ];
  for (const breakpoint of breakpoints) {
    if (width.value >= breakpoint.minWidth) {
      return Math.floor((width.value - 60) / breakpoint.columns);
    }
  }
  return 400;
});

// ==================== 展示模式 ====================
type DisplayMode = "list" | "vertical" | "horizontal" | "grid";
const displayMode = ref<DisplayMode>("vertical");

// ==================== 选择相关 ====================
const selectedIds = ref<number[]>([]);
// 瀑布流模式下的选择模式开关（列表/网格模式使用复选框，无需此开关）
const selectionMode = ref<boolean>(false);

const isAllSelected = computed(
  () =>
    images.value.length > 0 &&
    images.value.every((img) => selectedIds.value.includes(Number(img.id)))
);

function toggleSelection(id: number) {
  const idx = selectedIds.value.indexOf(id);
  if (idx >= 0) {
    selectedIds.value.splice(idx, 1);
  } else {
    selectedIds.value.push(id);
  }
}

function toggleSelectAll(val: any) {
  if (val) {
    selectedIds.value = images.value.map((img) => Number(img.id));
  } else {
    selectedIds.value = [];
  }
}

function selectAll() {
  selectedIds.value = images.value.map((img) => Number(img.id));
}

function clearSelection() {
  selectedIds.value = [];
}

// ==================== 图片数据派生 ====================
function extractImagesFromItem(item: DatasetItemVO): ImageUrlVO[] {
  const allImages: ImageUrlVO[] = [];
  if (item.clearImage) {
    allImages.push(item.clearImage);
  }
  if (item.hazyImages && item.hazyImages.length > 0) {
    allImages.push(...item.hazyImages);
  }
  return allImages;
}

/**
 * 判断图片是否已标注（hazeLevel 非空视为已标注）。
 * 适用于按"已标注/未标注"过滤。
 */
function isImageAnnotated(img: ImageUrlVO | undefined): boolean {
  return Boolean(img && img.hazeLevel);
}

/**
 * 从数据项中按标注过滤取代表图片：
 * - annotated：返回第一张 hazeLevel 非空的图片
 * - unannotated：返回第一张 hazeLevel 为空的图片
 * 都没有则回退到第一张图片。
 */
function pickImageByAnnotation(
  item: DatasetItemVO,
  filter: AnnotationFilter
): ImageUrlVO | undefined {
  const all = extractImagesFromItem(item);
  if (all.length === 0) return undefined;
  const matched =
    filter === "annotated"
      ? all.find((img) => isImageAnnotated(img))
      : all.find((img) => !isImageAnnotated(img));
  return matched || all[0];
}

// 当前展示的图片卡片（按"已标注/未标注"过滤派生）
const images = computed<ViewCard[]>(() => {
  return imageData
    .map((item) => {
      const img = pickImageFromItem(item);
      if (!img) return null;
      const isSelected = selectedIds.value.includes(item.id);
      return {
        id: item.id,
        src: img.url,
        originSrc: img.originUrl || img.url,
        alt: img.description || item.name,
        backgroundColor: isSelected ? "#ecf5ff" : "#fff",
      } as ViewCard;
    })
    .filter((v): v is ViewCard => v !== null);
});

/**
 * 根据当前 annotationFilter 从数据项中选取代表图片。
 * 仅返回符合当前过滤条件的图片，若数据项中无符合的图片则返回 null（不在列表中展示）。
 */
function pickImageFromItem(item: DatasetItemVO): ImageUrlVO | undefined {
  const all = extractImagesFromItem(item);
  if (all.length === 0) return undefined;
  const filtered =
    annotationFilter.value === "annotated"
      ? all.filter((img) => isImageAnnotated(img))
      : all.filter((img) => !isImageAnnotated(img));
  return filtered[0];
}

async function handleQuery() {
  queryParams.datasetId = datasetId.value;
  queryParams.pageNum = 1;
  imageData.length = 0;
  selectedIds.value = [];
  renderCount.value = 0;
  await loadMore();
}

async function loadMore() {
  queryParams.datasetId = datasetId.value;
  DatasetItemAPI.getList(queryParams)
    .then((data) => {
      const records = data.list || [];
      imageData.push(...records);
      total.value = data.total || 0;
      totalPages.value = Math.ceil(total.value / (queryParams.pageSize ?? 1));
    })
    .catch((err) => {
      console.log(err);
    });
}

function resetQuery() {
  queryParams.keyword = undefined;
  queryParams.sceneType = undefined;
  queryParams.hazeLevel = undefined;
  handleQuery();
}

/** 切换标注状态过滤 */
function handleAnnotationFilterChange(filter: AnnotationFilter) {
  annotationFilter.value = filter;
  selectedIds.value = [];
}

/** 获取图片类型标签（用于详情弹窗按钮文案） */
function getImageTypeLabel(type: string, hazeLevel?: string): string {
  const label = IMAGE_TYPE_LABELS[type] || type;
  if (hazeLevel) {
    const hazeLabel = formatHazeLevel(hazeLevel);
    return hazeLabel ? `${label}-${hazeLabel}` : label;
  }
  return label;
}

// ==================== 图片点击与详情弹窗 ====================
const detailDialogVisible = ref<boolean>(false);
const detailIndex = ref<number>(0);
const detailImageIndex = ref<number>(0);

const detailItem = computed(() => imageData[detailIndex.value]);
const detailImages = computed<ImageUrlVO[]>(() => {
  const item = imageData[detailIndex.value];
  return item ? extractImagesFromItem(item) : [];
});
const detailImage = computed(() => detailImages.value[detailImageIndex.value]);
const detailImageUrl = computed(() => {
  const img = detailImage.value;
  if (!img) return "";
  return img.originUrl || img.url;
});

function handleImageClick(itemId: number) {
  // 瀑布流模式下开启选择模式时，点击切换选中
  if (
    selectionMode.value &&
    (displayMode.value === "vertical" || displayMode.value === "horizontal")
  ) {
    toggleSelection(itemId);
    return;
  }
  openDetail(itemId);
}

function openDetail(itemId: number) {
  const idx = imageData.findIndex((item) => item.id === itemId);
  if (idx >= 0) {
    detailIndex.value = idx;
    detailImageIndex.value = 0;
    detailDialogVisible.value = true;
  }
}

function prevDetail() {
  if (detailIndex.value > 0) {
    detailIndex.value--;
    detailImageIndex.value = 0;
  }
}

function nextDetail() {
  if (detailIndex.value < imageData.length - 1) {
    detailIndex.value++;
    detailImageIndex.value = 0;
  }
}

function prevDetailImage() {
  if (detailImageIndex.value > 0) detailImageIndex.value--;
}

function nextDetailImage() {
  if (detailImageIndex.value < detailImages.value.length - 1)
    detailImageIndex.value++;
}

function handleDetailKeydown(e: KeyboardEvent) {
  if (e.key === "ArrowLeft") prevDetail();
  else if (e.key === "ArrowRight") nextDetail();
}

watch(detailDialogVisible, (val) => {
  if (val) {
    window.addEventListener("keydown", handleDetailKeydown);
  } else {
    window.removeEventListener("keydown", handleDetailKeydown);
  }
});

async function downloadDetailImage() {
  const item = detailItem.value;
  const img = detailImage.value;
  if (!item || !img) return;
  try {
    ElMessage.warning("下载功能请使用数据集整体下载（DatasetAPI.download）");
  } catch (err) {
    ElMessage.error("创建下载任务失败");
  }
}

async function deleteDetailItem() {
  const item = detailItem.value;
  if (!item) return;
  try {
    await ElMessageBox.confirm(
      `确定要删除「${item.name}」吗？此操作不可恢复！`,
      "警告",
      { type: "warning" }
    );
  } catch {
    return;
  }
  try {
    await DatasetItemAPI.deleteById(item.id);
    ElMessage.success("删除成功");
    const removeIdx = imageData.findIndex((d) => d.id === item.id);
    if (removeIdx >= 0) imageData.splice(removeIdx, 1);
    selectedIds.value = selectedIds.value.filter((id) => id !== item.id);
    if (imageData.length === 0) {
      detailDialogVisible.value = false;
    } else if (detailIndex.value >= imageData.length) {
      detailIndex.value = imageData.length - 1;
      detailImageIndex.value = 0;
    }
  } catch (err) {
    ElMessage.error("删除失败");
  }
}

// ==================== 列表/网格辅助 ====================
function getThumbUrl(row: DatasetItemVO): string {
  const img = row.clearImage || row.hazyImages?.[0];
  return img ? img.thumbnailUrl || img.url : "";
}

function getResolution(row: DatasetItemVO): string {
  const img = row.clearImage;
  return img && img.width && img.height ? `${img.width}×${img.height}` : "-";
}

function formatTime(t?: Date | string): string {
  if (!t) return "-";
  return new Date(t).toLocaleString("zh-CN");
}

async function downloadItem(row: DatasetItemVO) {
  const fileIds = extractImagesFromItem(row).map((img) => img.id);
  try {
    ElMessage.warning("单条数据集项下载功能暂未开放，请使用数据集整体下载");
  } catch (err) {
    ElMessage.error("下载失败");
  }
}

async function deleteItem(row: DatasetItemVO) {
  try {
    await ElMessageBox.confirm(
      `确定要删除「${row.name}」吗？此操作不可恢复！`,
      "警告",
      { type: "warning" }
    );
  } catch {
    return;
  }
  try {
    await DatasetItemAPI.deleteById(row.id);
    ElMessage.success("删除成功");
    const idx = imageData.findIndex((d) => d.id === row.id);
    if (idx >= 0) imageData.splice(idx, 1);
    selectedIds.value = selectedIds.value.filter((id) => id !== row.id);
  } catch (err) {
    ElMessage.error("删除失败");
  }
}

// ==================== 批量下载/删除 ====================
async function handleBatchDownload() {
  if (selectedIds.value.length === 0) {
    ElMessage.warning("请先选择要下载的图片");
    return;
  }
  const itemFileIds: number[] = [];
  selectedIds.value.forEach((id) => {
    const item = imageData.find((d) => d.id === id);
    if (item) {
      extractImagesFromItem(item).forEach((img) => itemFileIds.push(img.id));
    }
  });
  try {
    ElMessage.warning("批量下载功能暂未开放，请使用数据集整体下载");
  } catch (err) {
    ElMessage.error("下载失败");
  }
}

async function handleBatchDelete() {
  if (selectedIds.value.length === 0) {
    ElMessage.warning("请先选择要删除的图片");
    return;
  }
  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${selectedIds.value.length} 项数据吗？此操作不可恢复！`,
      "警告",
      { type: "warning" }
    );
  } catch {
    return;
  }
  try {
    const res = await DatasetItemAPI.batchDelete({ ids: selectedIds.value });
    ElMessage.success(`删除成功 ${res.successCount} 项`);
    selectedIds.value = [];
    handleQuery();
  } catch (err) {
    ElMessage.error("批量删除失败");
  }
}

// ==================== 上传弹窗 ====================
function handleUploadCommand(cmd: string) {
  if (cmd === "paired") {
    uploadDialogVisible.value = true;
  } else if (cmd === "batch") {
    batchUploadDialogVisible.value = true;
  }
}

// 配对上传
const uploadDialogVisible = ref<boolean>(false);
const clearFileList = ref<UploadUserFile[]>([]);
const hazyFileList = ref<UploadUserFile[]>([]);
const hazeLevel = ref<string>("");
const pairedSceneType = ref<string>("");
const uploading = ref<boolean>(false);

function resetPairedUpload() {
  clearFileList.value = [];
  hazyFileList.value = [];
  hazeLevel.value = "";
  pairedSceneType.value = "";
}

async function submitPairedUpload() {
  // 清晰图和有雾图均为可选（适配不同数据集规范），但至少上传一张图片
  if (clearFileList.value.length === 0 && hazyFileList.value.length === 0) {
    ElMessage.warning("请至少上传一张图片（清晰图或有雾图）");
    return;
  }
  const formData = new FormData();
  formData.append("datasetId", String(datasetId.value));
  if (pairedSceneType.value) {
    formData.append("sceneType", pairedSceneType.value);
  }
  if (clearFileList.value.length > 0 && clearFileList.value[0].raw) {
    formData.append("clearImage", clearFileList.value[0].raw as Blob);
  }
  if (hazyFileList.value.length > 0) {
    hazyFileList.value.forEach((f) => {
      formData.append("hazyImages", f.raw as Blob);
      // 每张有雾图对应一个雾霾程度（支持空字符串表示未标注）
      formData.append("hazeLevels", hazeLevel.value);
    });
  }
  uploading.value = true;
  try {
    await DatasetItemAPI.uploadImagePair(formData);
    ElMessage.success("上传成功");
    uploadDialogVisible.value = false;
    resetPairedUpload();
    handleQuery();
  } catch (err) {
    ElMessage.error("上传失败");
  } finally {
    uploading.value = false;
  }
}

// 批量上传
const batchUploadDialogVisible = ref<boolean>(false);
const batchFileList = ref<UploadUserFile[]>([]);
const batchSceneType = ref<string>("");
const batchUploading = ref<boolean>(false);

function resetBatchUpload() {
  batchFileList.value = [];
  batchSceneType.value = "";
}

async function submitBatchUpload() {
  if (batchFileList.value.length === 0) {
    ElMessage.warning("请选择文件");
    return;
  }
  const formData = new FormData();
  formData.append("datasetId", String(datasetId.value));
  if (batchSceneType.value) {
    formData.append("sceneType", batchSceneType.value);
  }
  batchFileList.value.forEach((f) => {
    formData.append("files", f.raw as Blob);
  });
  batchUploading.value = true;
  try {
    const res = await DatasetItemAPI.batchUpload(formData);
    ElMessage.success(
      `上传完成：成功 ${res.succeeded} 项，失败 ${res.failed} 项`
    );
    batchUploadDialogVisible.value = false;
    resetBatchUpload();
    handleQuery();
  } catch (err) {
    ElMessage.error("批量上传失败");
  } finally {
    batchUploading.value = false;
  }
}

// ==================== 统计分析弹窗 ====================
const statisticsDialogVisible = ref<boolean>(false);
const sceneChartRef = ref<HTMLDivElement>();
const hazeChartRef = ref<HTMLDivElement>();
const formatChartRef = ref<HTMLDivElement>();
const resolutionChartRef = ref<HTMLDivElement>();
let chartInstances: echarts.ECharts[] = [];

// 分辨率分布（从当前已加载数据派生，遍历所有图片而非仅 clearImage）
const resolutionDistribution = computed<Record<string, number>>(() => {
  const dist: Record<string, number> = {};
  imageData.forEach((item) => {
    extractImagesFromItem(item).forEach((img) => {
      if (img.width && img.height) {
        const key = `${img.width}×${img.height}`;
        dist[key] = (dist[key] || 0) + 1;
      }
    });
  });
  return dist;
});

function disposeCharts() {
  chartInstances.forEach((c) => c.dispose());
  chartInstances = [];
}

function handleChartResize() {
  chartInstances.forEach((c) => c.resize());
}

function initStatisticsCharts() {
  nextTick(() => {
    disposeCharts();
    const stats = datasetInfo.value.statistics;

    // 场景类型分布饼图
    if (sceneChartRef.value) {
      const chart = markRaw(echarts.init(sceneChartRef.value));
      const data = stats
        ? Object.entries(stats.sceneDistribution).map(([name, value]) => ({
            name,
            value,
          }))
        : [];
      chart.setOption({
        title: { text: "场景类型分布", left: "center" },
        tooltip: { trigger: "item" },
        legend: { bottom: 0 },
        series: [{ type: "pie", radius: "50%", data }],
      });
      chartInstances.push(chart);
    }

    // 雾霾程度分布柱状图
    if (hazeChartRef.value) {
      const chart = markRaw(echarts.init(hazeChartRef.value));
      const dist = stats?.hazeDistribution || {};
      const keys = Object.keys(dist);
      chart.setOption({
        title: { text: "雾霾程度分布", left: "center" },
        tooltip: { trigger: "axis" },
        xAxis: {
          type: "category",
          data: keys.map((k) => formatHazeLevel(k) || k),
        },
        yAxis: { type: "value" },
        series: [{ type: "bar", data: Object.values(dist) }],
      });
      chartInstances.push(chart);
    }

    // 文件格式分布饼图
    if (formatChartRef.value) {
      const chart = markRaw(echarts.init(formatChartRef.value));
      const data = stats
        ? Object.entries(stats.formatDistribution).map(([name, value]) => ({
            name,
            value,
          }))
        : [];
      chart.setOption({
        title: { text: "文件格式分布", left: "center" },
        tooltip: { trigger: "item" },
        legend: { bottom: 0 },
        series: [{ type: "pie", radius: "50%", data }],
      });
      chartInstances.push(chart);
    }

    // 分辨率分布柱状图
    if (resolutionChartRef.value) {
      const chart = markRaw(echarts.init(resolutionChartRef.value));
      const dist = resolutionDistribution.value;
      const keys = Object.keys(dist);
      chart.setOption({
        title: { text: "分辨率分布", left: "center" },
        tooltip: { trigger: "axis" },
        xAxis: {
          type: "category",
          data: keys,
          axisLabel: { rotate: 30 },
        },
        yAxis: { type: "value" },
        series: [{ type: "bar", data: Object.values(dist) }],
      });
      chartInstances.push(chart);
    }
  });
}

// ==================== 生命周期 ====================
onMounted(async () => {
  const id = Number(route.params.id);
  if (!id || isNaN(id)) {
    ElMessage.error("数据集ID无效，请从数据集列表进入");
    return;
  }
  datasetId.value = id;
  queryParams.datasetId = datasetId.value;
  await DatasetAPI.getDatasetInfoById(datasetId.value).then((data) => {
    datasetInfo.value = data;
  });
  await handleQuery();
  loadingObserver.value = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting && queryParams.pageNum! < totalPages.value) {
        queryParams.pageNum = queryParams.pageNum! + 1;
        loadMore();
      }
    });
  });

  if (loadingBarRef.value) {
    let loadingBarEl = loadingBarRef.value.$el as HTMLElement;
    loadingBarEl.style.transform = "translate3d(0, 3000px, 0)";
    loadingObserver.value.observe(loadingBarEl);
    setTimeout(() => (loadingBarEl.style.transform = "none"), 1000);
  }

  window.addEventListener("resize", handleChartResize);
});

onUnmounted(() => {
  loadingObserver.value?.disconnect();
  window.removeEventListener("resize", handleChartResize);
  window.removeEventListener("keydown", handleDetailKeydown);
  disposeCharts();
});
</script>

<template>
  <div class="app-container">
    <el-card shadow="never">
      <!-- 头部信息 -->
      <h1 class="mt-2 mb-3" style="text-align: center">
        {{ datasetInfo.name }} {{ datasetInfo.type }}数据集
      </h1>
      <p class="mr-3 ml-3 mb-6" style="text-indent: 2em">
        {{ datasetInfo.description }}
      </p>

      <!-- 统计摘要 + 统计分析按钮 -->
      <div class="summary-bar mb-3">
        <div class="summary-tags">
          <el-tag>图片总数：{{ datasetInfo.total || total }}</el-tag>
          <el-tag type="success" v-if="datasetInfo.statistics">
            已标注：{{ datasetInfo.statistics.annotatedCount }}
          </el-tag>
          <el-tag type="warning" v-if="datasetInfo.statistics">
            未标注：{{ datasetInfo.statistics.unannotatedCount }}
          </el-tag>
          <el-tag type="info" v-if="datasetInfo.statistics">
            文件总数：{{ datasetInfo.statistics.fileCount }}
          </el-tag>
        </div>
        <el-button type="primary" plain @click="statisticsDialogVisible = true">
          <template #icon>
            <el-icon><DataAnalysis /></el-icon>
          </template>
          统计分析
        </el-button>
      </div>

      <!-- 工具栏 -->
      <div class="toolbar mb-3">
        <div class="toolbar-left">
          <!-- 展示模式切换 -->
          <el-button-group>
            <el-button
              :type="displayMode === 'list' ? 'primary' : ''"
              size="small"
              title="列表模式"
              @click="displayMode = 'list'"
            >
              <el-icon><List /></el-icon>
            </el-button>
            <el-button
              :type="displayMode === 'vertical' ? 'primary' : ''"
              size="small"
              title="纵向瀑布流"
              @click="displayMode = 'vertical'"
            >
              <el-icon><Menu /></el-icon>
            </el-button>
            <el-button
              :type="displayMode === 'horizontal' ? 'primary' : ''"
              size="small"
              title="横向瀑布流"
              @click="displayMode = 'horizontal'"
            >
              <el-icon><Sort /></el-icon>
            </el-button>
            <el-button
              :type="displayMode === 'grid' ? 'primary' : ''"
              size="small"
              title="网格模式"
              @click="displayMode = 'grid'"
            >
              <el-icon><Grid /></el-icon>
            </el-button>
          </el-button-group>

          <!-- 标注状态切换：已标注/未标注 -->
          <el-button-group>
            <el-button
              :type="annotationFilter === 'annotated' ? 'primary' : ''"
              plain
              size="small"
              @click="handleAnnotationFilterChange('annotated')"
            >
              已标注
            </el-button>
            <el-button
              :type="annotationFilter === 'unannotated' ? 'primary' : ''"
              plain
              size="small"
              @click="handleAnnotationFilterChange('unannotated')"
            >
              未标注
            </el-button>
          </el-button-group>

          <!-- 瀑布流选择模式开关 -->
          <el-switch
            v-if="displayMode === 'vertical' || displayMode === 'horizontal'"
            v-model="selectionMode"
            active-text="选择模式"
            inline-prompt
          />
        </div>

        <div class="toolbar-right">
          <el-form :inline="true" :model="queryParams" size="small">
            <el-form-item>
              <el-input
                v-model="queryParams.keyword"
                clearable
                placeholder="图片名称"
                @keyup.enter="handleQuery"
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="handleQuery">
                <el-icon><Search /></el-icon>
                搜索
              </el-button>
              <el-button @click="resetQuery">
                <el-icon><Refresh /></el-icon>
                重置
              </el-button>
            </el-form-item>
          </el-form>

          <el-button-group>
            <el-button size="small" @click="selectAll">全选</el-button>
            <el-button
              size="small"
              :disabled="selectedIds.length === 0"
              @click="clearSelection"
            >
              清空
            </el-button>
          </el-button-group>

          <!-- 上传下拉 -->
          <el-dropdown @command="handleUploadCommand">
            <el-button type="primary" size="small">
              <el-icon><Upload /></el-icon>
              上传
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="paired">配对上传</el-dropdown-item>
                <el-dropdown-item command="batch">批量上传</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>

          <el-button
            type="success"
            size="small"
            :disabled="selectedIds.length === 0"
            @click="handleBatchDownload"
          >
            <el-icon><Download /></el-icon>
            下载({{ selectedIds.length }})
          </el-button>
          <el-button
            type="danger"
            size="small"
            :disabled="selectedIds.length === 0"
            @click="handleBatchDelete"
          >
            <el-icon><Delete /></el-icon>
            删除({{ selectedIds.length }})
          </el-button>
        </div>
      </div>

      <!-- 图片展示区域 -->
      <el-skeleton
        v-if="renderCount === 0 && datasetInfo.total !== 0"
        :rows="12"
        animated
      />

      <!-- 列表模式 -->
      <el-table
        v-if="displayMode === 'list'"
        :data="imageData as DatasetItemVO[]"
        row-key="id"
        border
        stripe
      >
        <el-table-column width="50" align="center">
          <template #header>
            <el-checkbox
              :model-value="isAllSelected"
              @change="toggleSelectAll"
            />
          </template>
          <template #default="{ row }">
            <el-checkbox
              :model-value="selectedIds.includes((row as DatasetItemVO).id)"
              @change="toggleSelection((row as DatasetItemVO).id)"
            />
          </template>
        </el-table-column>
        <el-table-column label="缩略图" width="100" align="center">
          <template #default="{ row }">
            <el-image
              :src="getThumbUrl(row as DatasetItemVO)"
              fit="cover"
              style="width: 80px; height: 60px; cursor: pointer"
              @click="openDetail((row as DatasetItemVO).id)"
            />
          </template>
        </el-table-column>
        <el-table-column
          label="文件名"
          prop="name"
          min-width="180"
          show-overflow-tooltip
        >
          <template #default="{ row }">
            <el-link
              type="primary"
              @click="openDetail((row as DatasetItemVO).id)"
            >
              {{ (row as DatasetItemVO).name }}
            </el-link>
          </template>
        </el-table-column>
        <el-table-column label="分辨率" width="120" align="center">
          <template #default="{ row }">{{
            getResolution(row as DatasetItemVO)
          }}</template>
        </el-table-column>
        <el-table-column label="文件大小" width="100" align="center">
          <template #default="{ row }">
            {{ (row as DatasetItemVO).clearImage?.formattedSize || "-" }}
          </template>
        </el-table-column>
        <el-table-column label="雾霾程度" width="110" align="center">
          <template #default="{ row }">
            <el-tag
              v-for="h in extractImagesFromItem(row as DatasetItemVO)"
              :key="h.id"
              size="small"
              type="warning"
              class="mr-1"
            >
              {{ formatHazeLevel(h.hazeLevel) || "未标注" }}
            </el-tag>
            <span
              v-if="extractImagesFromItem(row as DatasetItemVO).length === 0"
              >-</span
            >
          </template>
        </el-table-column>
        <el-table-column label="场景类型" width="120" align="center">
          <template #default="{ row }">{{
            (row as DatasetItemVO).sceneType || "-"
          }}</template>
        </el-table-column>
        <el-table-column label="图片数" width="80" align="center">
          <template #default="{ row }">
            {{
              (row as DatasetItemVO).imageCount ||
              extractImagesFromItem(row as DatasetItemVO).length
            }}
          </template>
        </el-table-column>
        <el-table-column label="上传时间" width="170" align="center">
          <template #default="{ row }">{{
            formatTime((row as DatasetItemVO).createTime)
          }}</template>
        </el-table-column>
        <el-table-column label="操作" width="160" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              link
              type="primary"
              @click="downloadItem(row as DatasetItemVO)"
            >
              下载
            </el-button>
            <el-button
              link
              type="danger"
              @click="deleteItem(row as DatasetItemVO)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 纵向瀑布流 -->
      <LongitudinalWaterfall
        v-else-if="displayMode === 'vertical'"
        :list="images"
        :width="itemWidth"
        @click-item="handleImageClick"
        @after-render="() => renderCount++"
      />

      <!-- 横向瀑布流 -->
      <Waterfall
        v-else-if="displayMode === 'horizontal'"
        :list="images"
        :width="itemWidth"
      />

      <!-- 网格模式 -->
      <el-row v-else :gutter="12">
        <el-col
          v-for="item in images"
          :key="item.id"
          :xs="12"
          :sm="8"
          :md="6"
          :lg="4"
          :xl="3"
          class="mb-3"
        >
          <el-card
            shadow="hover"
            body-style="padding: 8px"
            class="grid-card"
            @click="openDetail(Number(item.id))"
          >
            <div class="grid-checkbox" @click.stop>
              <el-checkbox
                :model-value="selectedIds.includes(Number(item.id))"
                @change="toggleSelection(Number(item.id))"
              />
            </div>
            <el-image :src="item.src" fit="cover" class="grid-image" />
            <div class="grid-info">
              <span class="grid-name" :title="item.alt">{{ item.alt }}</span>
            </div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 空状态 -->
      <el-empty
        v-if="images.length === 0 && renderCount > 0"
        description="暂无图片"
      />

      <!-- 加载更多 -->
      <el-divider
        v-show="
          totalPages > 1 &&
          renderCount >= queryParams.pageNum! - 1 &&
          queryParams.pageNum! < totalPages
        "
        ref="loadingBarRef"
      >
        正在加载，请稍后
      </el-divider>
    </el-card>

    <!-- 统计分析弹窗 -->
    <el-dialog
      v-model="statisticsDialogVisible"
      title="统计分析"
      width="900px"
      @open="initStatisticsCharts"
      @closed="disposeCharts"
    >
      <el-row :gutter="16">
        <el-col :span="12">
          <div ref="sceneChartRef" class="chart-box"></div>
        </el-col>
        <el-col :span="12">
          <div ref="hazeChartRef" class="chart-box"></div>
        </el-col>
        <el-col :span="12">
          <div ref="formatChartRef" class="chart-box"></div>
        </el-col>
        <el-col :span="12">
          <div ref="resolutionChartRef" class="chart-box"></div>
        </el-col>
      </el-row>
      <el-alert
        v-if="!datasetInfo.statistics"
        class="mt-3"
        type="warning"
        :closable="false"
        title="当前数据集暂无统计信息，部分图表可能为空"
      />
    </el-dialog>

    <!-- 配对上传弹窗 -->
    <el-dialog
      v-model="uploadDialogVisible"
      title="配对图片上传"
      width="800px"
      @closed="resetPairedUpload"
    >
      <el-form label-width="120px">
        <el-form-item label="清晰图像（可选）">
          <el-upload
            v-model:file-list="clearFileList"
            :auto-upload="false"
            :limit="1"
            accept="image/jpeg,image/png,image/gif"
            list-type="picture-card"
          >
            <el-icon><Plus /></el-icon>
          </el-upload>
        </el-form-item>
        <el-form-item label="有雾图像（可选）">
          <el-upload
            v-model:file-list="hazyFileList"
            :auto-upload="false"
            multiple
            accept="image/jpeg,image/png,image/gif"
            list-type="picture-card"
          >
            <el-icon><Plus /></el-icon>
          </el-upload>
        </el-form-item>
        <el-form-item label="雾霾程度（可选）">
          <el-input
            v-model="hazeLevel"
            placeholder="如 light/medium/heavy/beta=0.5"
            maxlength="50"
          />
        </el-form-item>
        <el-form-item label="场景类型">
          <el-input
            v-model="pairedSceneType"
            placeholder="请输入场景类型"
            maxlength="50"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="uploadDialogVisible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="uploading"
          @click="submitPairedUpload"
        >
          确认上传
        </el-button>
      </template>
    </el-dialog>

    <!-- 批量上传弹窗 -->
    <el-dialog
      v-model="batchUploadDialogVisible"
      title="批量上传"
      width="700px"
      @closed="resetBatchUpload"
    >
      <el-alert type="info" :closable="false" show-icon class="mb-3">
        <template #title>文件名自动识别配对规则</template>
        <div class="pairing-rules">
          xxx_clear.jpg / xxx_gt.jpg 识别为清晰图；xxx_hazy.jpg
          识别为有雾图；xxx_trans.jpg 识别为透射图；xxx_depth.jpg
          识别为深度图；xxx_segment.jpg 识别为分割图。雾霾程度
          支持多种规范：xxx_hazy_light.jpg / xxx_hazy_medium.jpg /
          xxx_hazy_heavy.jpg （人工分级），或
          xxx_1_0.74905.jpg（学术参数格式，统一取最后一个数值作为 beta）。
          同一前缀（前导数字）的图片自动归为一个配对组。
        </div>
      </el-alert>
      <el-upload
        v-model:file-list="batchFileList"
        :auto-upload="false"
        multiple
        accept="image/jpeg,image/png,image/gif"
        drag
      >
        <el-icon><UploadFilled /></el-icon>
        <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      </el-upload>
      <el-form label-width="120px" class="mt-3">
        <el-form-item label="场景类型">
          <el-input
            v-model="batchSceneType"
            placeholder="可选，应用于所有配对"
            maxlength="50"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="batchUploadDialogVisible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="batchUploading"
          @click="submitBatchUpload"
        >
          确认上传
        </el-button>
      </template>
    </el-dialog>

    <!-- 图片详情弹窗 -->
    <el-dialog
      v-model="detailDialogVisible"
      :title="detailItem?.name || '图片详情'"
      width="90vw"
      top="5vh"
    >
      <div class="detail-container">
        <!-- 图片查看区 -->
        <div class="detail-image-area">
          <el-image :src="detailImageUrl" fit="contain" class="detail-image" />
          <div class="detail-image-tabs" v-if="detailImages.length > 1">
            <el-button
              v-for="(img, idx) in detailImages"
              :key="img.id"
              :type="detailImageIndex === idx ? 'primary' : ''"
              size="small"
              @click="detailImageIndex = idx"
            >
              {{ getImageTypeLabel(img.type, img.hazeLevel) }}
            </el-button>
            <el-button
              size="small"
              :disabled="detailImageIndex <= 0"
              @click="prevDetailImage"
            >
              <el-icon><ArrowLeft /></el-icon>
            </el-button>
            <el-button
              size="small"
              :disabled="detailImageIndex >= detailImages.length - 1"
              @click="nextDetailImage"
            >
              <el-icon><ArrowRight /></el-icon>
            </el-button>
          </div>
        </div>

        <!-- 信息面板 -->
        <div class="detail-info-panel">
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="文件名">
              {{ detailImage?.fileName || detailItem?.name || "-" }}
            </el-descriptions-item>
            <el-descriptions-item label="格式">
              {{ detailImage?.format || "-" }}
            </el-descriptions-item>
            <el-descriptions-item label="分辨率">
              {{
                detailImage?.width && detailImage?.height
                  ? `${detailImage.width}×${detailImage.height}`
                  : "-"
              }}
            </el-descriptions-item>
            <el-descriptions-item label="文件大小">
              {{ detailImage?.formattedSize || "-" }}
            </el-descriptions-item>
            <el-descriptions-item label="雾霾程度">
              <el-tag v-if="detailImage?.hazeLevel" size="small" type="warning">
                {{ formatHazeLevel(detailImage.hazeLevel) }}
              </el-tag>
              <span v-else>-</span>
            </el-descriptions-item>
            <el-descriptions-item label="场景类型">
              {{ detailImage?.sceneType || detailItem?.sceneType || "-" }}
            </el-descriptions-item>
            <el-descriptions-item label="使用次数">
              {{ detailImage?.usageCount ?? "-" }}
            </el-descriptions-item>
            <el-descriptions-item label="上传时间">
              {{ formatTime(detailImage?.createTime) }}
            </el-descriptions-item>
          </el-descriptions>
          <div class="detail-actions mt-3">
            <el-button type="primary" @click="downloadDetailImage">
              <el-icon><Download /></el-icon>
              下载
            </el-button>
            <el-button type="danger" @click="deleteDetailItem">
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </div>
        </div>
      </div>

      <!-- 导航底栏 -->
      <div class="detail-nav">
        <el-button :disabled="detailIndex <= 0" @click="prevDetail">
          <el-icon><ArrowLeft /></el-icon>
          上一张
        </el-button>
        <span class="nav-position">
          {{ imageData.length === 0 ? 0 : detailIndex + 1 }} /
          {{ imageData.length }}
        </span>
        <el-button
          :disabled="detailIndex >= imageData.length - 1"
          @click="nextDetail"
        >
          下一张
          <el-icon><ArrowRight /></el-icon>
        </el-button>
      </div>
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.summary-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;

  .summary-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: space-between;

  .toolbar-left,
  .toolbar-right {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
}

/* 网格模式 */
.grid-card {
  position: relative;
  cursor: pointer;

  .grid-checkbox {
    position: absolute;
    top: 4px;
    left: 4px;
    z-index: 2;
    padding: 2px;
    background: rgb(255 255 255 / 80%);
    border-radius: 4px;
  }

  .grid-image {
    display: block;
    width: 100%;
    height: 180px;
  }

  .grid-info {
    padding: 6px 2px 0;

    .grid-name {
      display: inline-block;
      width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 13px;
      color: #606266;
      white-space: nowrap;
    }
  }
}

/* 统计图表 */
.chart-box {
  width: 100%;
  height: 300px;
}

.pairing-rules {
  font-size: 13px;
  line-height: 1.6;
}

/* 详情弹窗 */
.detail-container {
  display: flex;
  gap: 16px;
  height: 70vh;

  .detail-image-area {
    display: flex;
    flex: 1;
    flex-direction: column;
    align-items: center;
    padding: 12px;
    background: #f5f7fa;
    border-radius: 4px;

    .detail-image {
      max-width: 100%;
      max-height: calc(70vh - 60px);
    }

    .detail-image-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: center;
      margin-top: 12px;
    }
  }

  .detail-info-panel {
    flex-shrink: 0;
    width: 320px;
    overflow-y: auto;

    .detail-actions {
      display: flex;
      gap: 8px;
    }
  }
}

.detail-nav {
  display: flex;
  gap: 16px;
  align-items: center;
  justify-content: center;
  margin-top: 16px;

  .nav-position {
    min-width: 60px;
    font-size: 14px;
    color: #606266;
    text-align: center;
  }
}
</style>
