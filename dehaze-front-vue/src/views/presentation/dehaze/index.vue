<script lang="ts" setup>
import AlgorithmToolBar from "@/components/AlgorithmToolBar/index.vue";
import Camera from "@/components/Camera/index.vue";
import DatasetImageSelect from "@/components/DatasetImageSelect/index.vue";
import ExampleImageSelect from "@/components/ExampleImageSelect/index.vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import SingleImageShow from "@/components/SingleImageShow/index.vue";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useAlgorithmStore } from "@/store";
import { useImageShowStore } from "@/store/modules/imageShow";
import examples from "@/views/presentation/dehaze/exampleImages";
import { FileAPI, ModelAPI } from "dehaze-sdk-js";
import { UploadFile, UploadUserFile } from "element-plus";

const algorithmStore = useAlgorithmStore();
const imageShowStore = useImageShowStore();

const { imageInfo } = toRefs(imageShowStore);
const { images } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);

const exampleHazeUrls = computed(() => examples.map((item) => item.haze));
const cleanUrl = ref("");
const selectedModel = ref<number>();

// 当前激活的页面（单值状态机，替代多个布尔标志）
type PageName =
  "camera" | "singleImage" | "example" | "overlap" | "loading" | "batch";
const activePage = ref<PageName>("example");

const disableMore = computed(() => activePage.value !== "overlap");

// ============ 去雾算法参数（来自 AlgorithmToolBar） ============
const dehazeParams = ref({
  dehazeStrength: 50,
  colorSaturation: 50,
  contrast: 50,
  sharpen: 30,
});

function handleParamChange(params: typeof dehazeParams.value) {
  dehazeParams.value = params;
}

// ============ 处理进度显示（5阶段） ============
// 预处理(0-10)、算法初始化(10-20)、去雾处理(20-90)、后处理(90-95)、保存(95-100)
const stages = [
  { name: "预处理", min: 0, max: 10 },
  { name: "算法初始化", min: 10, max: 20 },
  { name: "去雾处理", min: 20, max: 90 },
  { name: "后处理", min: 90, max: 95 },
  { name: "保存", min: 95, max: 100 },
];
const progress = ref(0);
const processing = ref(false);
// 取消标志位，取消后忽略后续回调
let cancelFlag = false;
let progressTimer: number | undefined;

// 当前所处阶段索引
const currentStageIndex = computed(() => {
  for (let i = 0; i < stages.length; i++) {
    if (progress.value < stages[i].max) return i;
  }
  return stages.length - 1;
});

// 启动模拟进度（定时器递增），cap 为暂停上限（等待后端真实完成时跳到 100）
function startProgressSimulation(cap = 95) {
  stopProgressSimulation();
  progressTimer = window.setInterval(() => {
    if (progress.value >= cap) return;
    // 在当前阶段范围内随机递增
    const step = Math.random() * 4 + 1;
    progress.value = Math.min(progress.value + step, cap);
  }, 300);
}

function stopProgressSimulation() {
  if (progressTimer !== undefined) {
    clearInterval(progressTimer);
    progressTimer = undefined;
  }
}

function handleCameraSave(file: File) {
  // 上传文件
  handleImageUpload(file);
}

function handleSelectModel(id: number) {
  imageShowStore.setModelId(id);
}

function handleImageUpload(file: File) {
  imageShowStore.setLoading(true);
  // 上传文件
  FileAPI.upload(file, imageShowStore.modelId)
    .then((res) => {
      // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
      imageShowStore.setImageUrl(res.url, ImageTypeEnum.HAZE);
      activePage.value = "singleImage";
    })
    .catch((err) => {
      ElMessage.error(err.message || "图片上传失败");
    })
    .finally(() => {
      imageShowStore.setLoading(false);
    });
}

function handleReset() {
  imageShowStore.setImageUrls([]);
  imageShowStore.setMagnifierShow(false);
  cleanUrl.value = "";
  activePage.value = "example";
}

// 取消处理
function handleCancelProcess() {
  cancelFlag = true;
  processing.value = false;
  stopProgressSimulation();
  progress.value = 0;
  activePage.value = "singleImage";
  ElMessage.info("已取消处理");
}

// 选择模型后生成对比图（原图 | 去雾图），含5阶段进度显示
async function handleGenerateImage() {
  if (!selectedModel.value) {
    ElMessage.error("请选择去雾模型");
    return;
  }
  if (!imgUrls.value[0]) {
    ElMessage.error("请先上传图片");
    return;
  }
  // 显示确认对话框
  const modelOption = algorithmStore.algorithmOptions.find(
    (m: any) => m.value === selectedModel.value
  );
  const modelName = modelOption?.label || `模型ID: ${selectedModel.value}`;
  const imgUrl = imgUrls.value[0].url;
  const imageName = imgUrl.split("/").pop() || "未命名图片";
  try {
    await ElMessageBox.confirm(
      [
        `图片：${imageName}`,
        `算法：${modelName}`,
        `参数：去雾强度 ${dehazeParams.value.dehazeStrength}% / 色彩饱和度 ${dehazeParams.value.colorSaturation}% / 对比度 ${dehazeParams.value.contrast}% / 锐化 ${dehazeParams.value.sharpen}%`,
        "",
        "确认开始去雾处理？处理期间请勿离开页面。",
      ].join("\n"),
      "去雾处理确认",
      {
        confirmButtonText: "开始处理",
        cancelButtonText: "取消",
        type: "info",
        distinguishCancelAndClose: true,
      }
    );
  } catch (action) {
    if (action === "cancel") {
      ElMessage.info("已取消处理");
    }
    return;
  }
  const modelId = selectedModel.value;
  cancelFlag = false;
  processing.value = true;
  progress.value = 0;
  imageShowStore.setLoading(true);
  imageShowStore.setModelId(modelId);
  activePage.value = "loading";
  // 启动模拟进度
  startProgressSimulation(95);
  ModelAPI.predictAndWait({
    algorithmId: modelId,
    imageUrl: imgUrl,
    params: dehazeParams.value ? JSON.stringify(dehazeParams.value) : undefined,
  })
    .then(async (res) => {
      if (cancelFlag) return;
      if (res.status === "failed") {
        throw new Error(res.errorMessage || "去雾处理失败");
      }
      progress.value = 95;
      stopProgressSimulation();
      imageShowStore.setImageUrl(imgUrl, ImageTypeEnum.HAZE);
      imageShowStore.setImageUrl(res.resultUrl || "", ImageTypeEnum.PRED);
      if (cleanUrl.value) {
        try {
          const cleanRes = await handleCleanUrl(cleanUrl.value, modelId);
          cleanUrl.value = cleanRes;
        } catch (e: any) {
          ElMessage.error("清晰图上传失败：" + (e.message || "未知错误"));
        }
      }
      progress.value = 100;
      activePage.value = "overlap";
    })
    .catch((err) => {
      if (cancelFlag) return;
      ElMessage.error(err.message || "去雾处理失败");
      activePage.value = "singleImage";
    })
    .finally(() => {
      stopProgressSimulation();
      processing.value = false;
      imageShowStore.setLoading(false);
    });
}

// ============ 结果保存 ============
const saving = ref(false);
async function handleSaveResult() {
  const predImg = imgUrls.value.find(
    (img) => img.label.text === ImageTypeEnum.PRED
  );
  if (!predImg) {
    ElMessage.error("没有可保存的结果");
    return;
  }
  if (!selectedModel.value) {
    ElMessage.error("请先选择去雾模型");
    return;
  }
  saving.value = true;
  try {
    const res = await fetch(predImg.url);
    const blob = await res.blob();
    const file = new File([blob], `dehaze_result_${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
    await FileAPI.upload(file, selectedModel.value);
    ElMessage.success("结果保存成功");
  } catch (e: any) {
    ElMessage.error("保存失败：" + (e.message || "未知错误"));
  } finally {
    saving.value = false;
  }
}

// ============ 批量处理 ============
interface BatchTask {
  id: number;
  name: string;
  file: File;
  status: "pending" | "processing" | "success" | "failed" | "cancelled";
  progress: number;
  hazeUrl: string;
  resultUrl: string;
  errorMsg: string;
}
const batchTasks = ref<BatchTask[]>([]);
const batchProcessing = ref(false);
const batchCancelled = ref(false);
let batchIdCounter = 0;
let batchTimer: number | undefined;

// 批量整体进度
const batchOverallProgress = computed(() => {
  if (batchTasks.value.length === 0) return 0;
  const total = batchTasks.value.reduce((sum, t) => sum + t.progress, 0);
  return Math.round(total / batchTasks.value.length);
});

const batchDialogVisible = ref(false);
const batchUploadFiles = ref<UploadUserFile[]>([]);

function handleBatchFileChange(_file: UploadFile, files: UploadUserFile[]) {
  batchUploadFiles.value = files;
}

// 确认添加批量图片到任务列表
function handleConfirmBatchAdd() {
  const files = batchUploadFiles.value
    .map((f) => f.raw)
    .filter(Boolean) as File[];
  files.forEach((file) => {
    batchTasks.value.push({
      id: ++batchIdCounter,
      name: file.name,
      file,
      status: "pending",
      progress: 0,
      hazeUrl: "",
      resultUrl: "",
      errorMsg: "",
    });
  });
  batchUploadFiles.value = [];
  batchDialogVisible.value = false;
  activePage.value = "batch";
}

// 开始批量处理（串行）
async function handleStartBatch() {
  if (!selectedModel.value) {
    ElMessage.error("请选择去雾模型");
    return;
  }
  const modelId = selectedModel.value;
  batchCancelled.value = false;
  batchProcessing.value = true;
  imageShowStore.setModelId(modelId);
  for (const task of batchTasks.value) {
    if (task.status !== "pending") continue;
    if (batchCancelled.value) break;
    task.status = "processing";
    task.progress = 0;
    // 模拟单张进度
    batchTimer = window.setInterval(() => {
      if (task.progress >= 95) return;
      task.progress = Math.min(task.progress + Math.random() * 4 + 1, 95);
    }, 300);
    try {
      // 先上传原图
      const uploadRes = await FileAPI.upload(task.file, modelId);
      task.hazeUrl = uploadRes.url;
      const predRes = await ModelAPI.predictAndWait({
        algorithmId: modelId,
        imageUrl: task.hazeUrl,
        params: dehazeParams.value
          ? JSON.stringify(dehazeParams.value)
          : undefined,
      });
      if (predRes.status === "failed") {
        throw new Error(predRes.errorMessage || "处理失败");
      }
      task.resultUrl = predRes.resultUrl || "";
      task.progress = 100;
      task.status = "success";
    } catch (e: any) {
      task.status = "failed";
      task.errorMsg = e.message || "处理失败";
      task.progress = 0;
    } finally {
      if (batchTimer !== undefined) {
        clearInterval(batchTimer);
        batchTimer = undefined;
      }
    }
  }
  batchProcessing.value = false;
  if (batchCancelled.value) {
    ElMessage.info("批量处理已取消");
  } else {
    ElMessage.success("批量处理完成");
  }
}

// 取消批量处理
function handleCancelBatch() {
  batchCancelled.value = true;
  if (batchTimer !== undefined) {
    clearInterval(batchTimer);
    batchTimer = undefined;
  }
  batchTasks.value.forEach((task) => {
    if (task.status === "pending" || task.status === "processing") {
      task.status = "cancelled";
    }
  });
  batchProcessing.value = false;
}

// 重试失败的批量任务
function handleRetryBatch(task: BatchTask) {
  task.status = "pending";
  task.progress = 0;
  task.errorMsg = "";
}

// 移除批量任务
function handleRemoveBatch(task: BatchTask) {
  batchTasks.value = batchTasks.value.filter((t) => t.id !== task.id);
}

// 查看批量任务结果
function handleViewBatchResult(task: BatchTask) {
  imageShowStore.setImageUrls([]);
  if (task.hazeUrl) {
    imageShowStore.setImageUrl(task.hazeUrl, ImageTypeEnum.HAZE);
  }
  imageShowStore.setImageUrl(task.resultUrl, ImageTypeEnum.PRED);
  activePage.value = "overlap";
}

// 清空批量任务列表
function handleClearBatch() {
  batchTasks.value = [];
}

function statusTagColor(status: string): string {
  const map: Record<string, string> = {
    pending: "#1890ff",
    processing: "#1890ff",
    success: "#52c41a",
    failed: "#ff4d4f",
    cancelled: "#8c8c8c",
  };
  return map[status];
}

function statusText(status: string) {
  const map: Record<string, string> = {
    pending: "等待中",
    processing: "处理中",
    success: "已完成",
    failed: "已失败",
    cancelled: "已取消",
  };
  return map[status];
}

function handleExampleImageClick(url: string) {
  const matched = examples.find((item) => item.haze === url);
  if (!matched) return;
  imageShowStore.setImageUrl(url, ImageTypeEnum.HAZE);
  cleanUrl.value = matched.clean;
  activePage.value = "singleImage";
}

// 获取模型选项列表
async function getAlgorithmList() {
  await algorithmStore.getAlgorithmOptions();
}

function handleDatasetImageSelect(haze: string, clear: string) {
  imageShowStore.setImageUrl(haze, ImageTypeEnum.HAZE);
  cleanUrl.value = clear;
  dialogVisible.value = false;
  activePage.value = "singleImage";
}

async function handleCleanUrl(url: string, modelId: number) {
  const res = await fetch(url);
  const blob = await res.blob();
  const cleanFile = new File([blob], "clean.jpg", { type: "image/jpeg" });
  const cleanRes = await FileAPI.upload(cleanFile, modelId);
  return cleanRes.url;
}

const router = useRouter();
const route = useRoute();

function handleEval() {
  if (!selectedModel.value) {
    ElMessage.error("请先选择去雾模型");
    return;
  }
  router.push("/evaluation/index").then(() => {
    imageShowStore.setModelId(selectedModel.value!);
    imageShowStore.setImageUrls(imgUrls.value);
    if (cleanUrl.value) {
      imageShowStore.setImageUrl(cleanUrl.value, ImageTypeEnum.CLEAN);
    }
  });
}

const dialogVisible = ref(false);

// 从图像输入页跳转时，通过 query 参数 imageUrl 传入图片地址
function loadImageFromQuery() {
  const imageUrl = route.query.imageUrl as string;
  if (imageUrl) {
    imageShowStore.setImageUrls([]);
    imageShowStore.setImageUrl(imageUrl, ImageTypeEnum.HAZE);
    activePage.value = "singleImage";
    // 清除 query 参数，避免回退时重复加载
    router.replace({ path: "/presentation/dehaze" });
    return true;
  }
  return false;
}

onMounted(() => {
  imageShowStore.setLoading(true);
  getAlgorithmList();
  imageShowStore.setImageUrls([]);
  if (!loadImageFromQuery()) {
    activePage.value = "example";
  }
  imageShowStore.setLoading(false);
});

onActivated(() => {
  // keepAlive 缓存后再次激活时，检查是否携带 imageUrl
  loadImageFromQuery();
});

onUnmounted(() => {
  stopProgressSimulation();
  if (batchTimer !== undefined) {
    clearInterval(batchTimer);
  }
});
</script>

<template>
  <div class="app-container">
    <!-- 左侧工具栏 -->
    <AlgorithmToolBar
      :disable-more="disableMore"
      :show-dehaze-params="true"
      @on-upload="handleImageUpload"
      @on-eval="handleEval"
      @on-take-photo="activePage = 'camera'"
      @on-reset="handleReset"
      @on-generate="handleGenerateImage"
      @on-select-from-dataset="() => (dialogVisible = true)"
      @on-param-change="handleParamChange"
    >
      <!-- 选择模型区域 -->
      <template #default>
        <div class="select-wrap">
          <span>选择去雾模型</span>
          <el-tree-select
            v-model="selectedModel"
            :data="algorithmStore.algorithmOptions"
            placeholder="请选择去雾模型算法"
            style="width: 240px"
            @change="handleSelectModel"
          />
        </div>
        <div class="batch-entry-wrap">
          <el-button type="primary" plain @click="batchDialogVisible = true">
            批量处理
          </el-button>
          <el-button v-if="batchTasks.length > 0" @click="activePage = 'batch'">
            查看任务列表（{{ batchTasks.length }}）
          </el-button>
        </div>
      </template>
    </AlgorithmToolBar>
    <!-- 右侧功能栏 -->
    <el-card class="flex-center example-img-wrap">
      <!-- 样例图片显示 -->
      <ExampleImageSelect
        v-if="activePage === 'example'"
        :urls="exampleHazeUrls"
        class="example"
        @on-example-select="handleExampleImageClick"
      />
      <!-- 拍照上传 -->
      <Camera
        v-if="activePage === 'camera'"
        class="camera"
        @on-cancel="activePage = 'example'"
        @on-save="handleCameraSave"
      />
      <!-- 单图展示 -->
      <SingleImageShow
        v-if="activePage === 'singleImage'"
        :src="imgUrls[0].url || ''"
        class="single-image"
      />
      <!-- 处理进度显示（5阶段） -->
      <div v-if="activePage === 'loading'" class="progress-wrap">
        <h3 class="progress-title">正在处理图像</h3>
        <el-steps
          :active="currentStageIndex"
          align-center
          finish-status="success"
          class="progress-steps"
        >
          <el-step
            v-for="stage in stages"
            :key="stage.name"
            :title="stage.name"
          />
        </el-steps>
        <el-progress
          :percentage="progress"
          :stroke-width="22"
          :text-inside="true"
          status="success"
          class="progress-bar"
        />
        <p class="progress-stage-text">
          当前阶段：{{ stages[currentStageIndex].name }}（{{
            progress.toFixed(0)
          }}%）
        </p>
        <el-button
          :disabled="!processing"
          type="danger"
          @click="handleCancelProcess"
        >
          取消处理
        </el-button>
      </div>
      <!-- 重叠展示 -->
      <OverlapImageShow v-if="activePage === 'overlap'" class="overlap" />
      <!-- 处理完成后保存结果 -->
      <div v-if="activePage === 'overlap'" class="save-result-wrap">
        <el-button :loading="saving" type="success" @click="handleSaveResult">
          保存结果
        </el-button>
        <el-button type="primary" @click="handleEval"> 评估结果 </el-button>
      </div>
      <!-- 批量处理任务列表 -->
      <div v-if="activePage === 'batch'" class="batch-wrap">
        <div class="batch-header">
          <h3>批量去雾处理</h3>
          <div class="batch-actions">
            <el-button @click="batchDialogVisible = true">添加图片</el-button>
            <el-button
              :disabled="batchTasks.length === 0 || batchProcessing"
              type="primary"
              @click="handleStartBatch"
            >
              开始批量处理
            </el-button>
            <el-button
              v-if="batchProcessing"
              type="danger"
              @click="handleCancelBatch"
            >
              取消全部
            </el-button>
            <el-button :disabled="batchProcessing" @click="handleClearBatch">
              清空列表
            </el-button>
          </div>
        </div>
        <div v-if="batchTasks.length > 0" class="batch-overall">
          <span>整体进度</span>
          <el-progress
            :percentage="batchOverallProgress"
            :stroke-width="16"
            :text-inside="true"
          />
        </div>
        <el-table :data="batchTasks" border class="batch-table">
          <el-table-column label="图片名称" min-width="180" prop="name" />
          <el-table-column :width="100" align="center" label="状态">
            <template #default="scope">
              <el-tag :color="statusTagColor(scope.row.status)" effect="dark">
                {{ statusText(scope.row.status) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column :min-width="200" label="进度">
            <template #default="scope">
              <el-progress
                :percentage="scope.row.progress"
                :stroke-width="14"
                :text-inside="true"
                :status="
                  scope.row.status === 'failed'
                    ? 'exception'
                    : scope.row.status === 'success'
                      ? 'success'
                      : undefined
                "
              />
            </template>
          </el-table-column>
          <el-table-column
            :width="180"
            align="center"
            fixed="right"
            label="操作"
          >
            <template #default="scope">
              <el-button
                v-if="scope.row.status === 'success'"
                link
                type="primary"
                @click="handleViewBatchResult(scope.row as BatchTask)"
              >
                查看
              </el-button>
              <el-button
                v-if="scope.row.status === 'failed'"
                link
                type="warning"
                @click="handleRetryBatch(scope.row as BatchTask)"
              >
                重试
              </el-button>
              <el-button
                v-if="
                  scope.row.status === 'pending' ||
                  scope.row.status === 'failed'
                "
                link
                type="danger"
                @click="handleRemoveBatch(scope.row as BatchTask)"
              >
                移除
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <el-empty
          v-if="batchTasks.length === 0"
          description="暂无批量任务，请点击“添加图片”"
        />
      </div>
    </el-card>

    <!-- 批量上传对话框 -->
    <el-dialog
      v-model="batchDialogVisible"
      title="批量上传图片"
      top="8vh"
      width="50%"
    >
      <el-upload
        v-model:file-list="batchUploadFiles"
        :auto-upload="false"
        :limit="20"
        :multiple="true"
        :on-change="handleBatchFileChange"
        accept="image/gif, image/jpeg, image/jpg, image/png, image/svg"
        action="#"
        drag
      >
        <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
        <template #tip>
          <div class="el-upload__tip">
            支持多张图片，最多20张（gif/jpeg/jpg/png/svg）
          </div>
        </template>
      </el-upload>
      <template #footer>
        <el-button @click="batchDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleConfirmBatchAdd">
          添加到任务列表
        </el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="dialogVisible"
      title="选择数据集图片"
      top="8vh"
      width="70%"
    >
      <DatasetImageSelect @on-selected="handleDatasetImageSelect" />
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.app-container {
  display: flex;
  height: calc(100vh - $navbar-height);
}

.select-wrap {
  span {
    margin-right: 20px;
    font-size: 18px;
    font-weight: 700;
  }
}

.batch-entry-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  margin-top: 14px;
}

.flex-center {
  width: 64vw;
  overflow-y: auto;

  .example {
    padding-top: 100px;
  }

  .single-image {
    max-width: calc(64vw - 6vw);
    max-height: calc(100vh - $navbar-height - 40px - 20px - 10px);
  }

  .camera {
    width: calc(64vw - 6vw);
    height: calc(100vh - $navbar-height - 40px - 20px - 10px);
  }

  .overlap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: calc(64vw - 6vw);
    height: calc(100vh - $navbar-height - 40px - 20px - 10px);
    margin: 0 auto;
  }

  .progress-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: calc(64vw - 6vw);
    min-height: calc(100vh - $navbar-height - 40px - 20px - 10px);
    margin: 0 auto;

    .progress-title {
      margin-bottom: 30px;
      font-size: 22px;
      font-weight: 700;
      color: var(--el-color-primary);
    }

    .progress-steps {
      width: 100%;
      max-width: 700px;
      margin-bottom: 30px;
    }

    .progress-bar {
      width: 100%;
      max-width: 700px;
      margin-bottom: 16px;
    }

    .progress-stage-text {
      margin-bottom: 24px;
      font-size: 15px;
      color: #666;
    }
  }

  .save-result-wrap {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin: 16px 0;
  }

  .batch-wrap {
    width: calc(64vw - 4vw);
    margin: 20px auto;

    .batch-header {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;

      h3 {
        margin: 0;
      }

      .batch-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
    }

    .batch-overall {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 16px;
      font-size: 14px;
      color: #666;
    }

    .batch-table {
      margin-bottom: 12px;
    }
  }
}

@media screen and (width <=992px) {
  .app-container {
    display: flex;
    flex-wrap: wrap;
    height: auto;
  }

  .flex-center {
    width: 100vw;
    padding-top: 0;
    margin-top: 10px;

    .overlap {
      width: calc(100vw - 6vw);
      height: calc(100vh - $navbar-height - 40px);
    }
  }
}
</style>

<style lang="scss">
.example-img-wrap {
  .el-card__body {
    overflow: unset;
  }
}
</style>
