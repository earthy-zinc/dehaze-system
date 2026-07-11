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
import { changeUrl } from "@/utils";
import examples from "@/views/presentation/dehaze/exampleImages";
import { FileAPI, ModelAPI } from "dehaze-sdk-js";
import { UploadFile, UploadUserFile } from "element-plus";

const algorithmStore = useAlgorithmStore();
const imageShowStore = useImageShowStore();

const { imageInfo } = toRefs(imageShowStore);
const { images } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);

const exampleImages = ref(examples);
const exampleHazeUrls = computed(() =>
  exampleImages.value.map((item) => item.haze)
);
const cleanUrl = ref("");
const modelOptions = ref<OptionType[]>([]);
const selectedModel = ref<number>();

const show = reactive({
  camera: false,
  singleImage: false,
  example: false,
  loading: false,
  overlap: false,
  effect: true,
  batch: false,
});

const disableMore = computed(() => !show.overlap);

function activePage(
  page:
    | "camera"
    | "singleImage"
    | "example"
    | "overlap"
    | "loading"
    | "effect"
    | "batch"
) {
  show.camera = page === "camera";
  show.singleImage = page === "singleImage";
  show.example = page === "example";
  show.overlap = page === "overlap";
  show.loading = page === "loading";
  show.effect = page === "effect";
  show.batch = page === "batch";
}

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
      activePage("loading");
      imageShowStore.setImageUrl(changeUrl(res.url), ImageTypeEnum.HAZE);
    })
    .then(() => {
      // 将文件显示到 SingleImageShow 组件中
      activePage("singleImage");
    })
    .catch((err) => {
      ElMessage.error(err);
    })
    .finally(() => {
      imageShowStore.setLoading(false);
    });
}

function handleReset() {
  imageShowStore.setImageUrls([]);
  imageShowStore.toggleMagnifierShow();
  activePage("example");
}

// 取消处理
function handleCancelProcess() {
  cancelFlag = true;
  processing.value = false;
  stopProgressSimulation();
  progress.value = 0;
  activePage("singleImage");
  ElMessage.info("已取消处理");
}

// 选择模型后生成对比图（原图 | 去雾图），含5阶段进度显示
function handleGenerateImage() {
  if (!selectedModel.value) {
    ElMessage.error("请选择去雾模型");
    return;
  }
  if (!imgUrls.value[0]) {
    ElMessage.error("请先上传图片");
    return;
  }
  cancelFlag = false;
  processing.value = true;
  progress.value = 0;
  imageShowStore.setLoading(true);
  imageShowStore.setModelId(Number(selectedModel.value) || 1);
  activePage("loading");
  // 启动模拟进度
  startProgressSimulation(95);
  const imgUrl = imgUrls.value[0].url;
  ModelAPI.prediction({
    modelId: Number(selectedModel.value) || 1,
    url: imgUrl,
    modelParam: dehazeParams.value,
  })
    .then(async (res) => {
      if (cancelFlag) return;
      // 进入保存阶段
      progress.value = 95;
      stopProgressSimulation();
      // 获取生成后的图片url
      imageShowStore.setImageUrl(changeUrl(res.hazeUrl), ImageTypeEnum.HAZE);
      imageShowStore.setImageUrl(changeUrl(res.predUrl), ImageTypeEnum.PRED);
      if (cleanUrl.value) {
        const clean = cleanUrl.value;
        handleCleanUrl(clean).then(
          (cleanRes) => (cleanUrl.value = changeUrl(cleanRes))
        );
      }
      progress.value = 100;
      activePage("overlap");
    })
    .catch((err) => {
      if (cancelFlag) return;
      ElMessage.error(err);
      activePage("singleImage");
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
  saving.value = true;
  try {
    const res = await fetch(predImg.url);
    const blob = await res.blob();
    const file = new File([blob], `dehaze_result_${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
    await FileAPI.upload(file, Number(selectedModel.value) || 1);
    ElMessage.success("结果保存成功");
  } catch (e) {
    ElMessage.error("保存失败：" + String(e));
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
      resultUrl: "",
      errorMsg: "",
    });
  });
  batchUploadFiles.value = [];
  batchDialogVisible.value = false;
  activePage("batch");
}

// 开始批量处理（串行）
async function handleStartBatch() {
  if (!selectedModel.value) {
    ElMessage.error("请选择去雾模型");
    return;
  }
  batchCancelled.value = false;
  batchProcessing.value = true;
  imageShowStore.setModelId(Number(selectedModel.value) || 1);
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
      const uploadRes = await FileAPI.upload(
        task.file,
        Number(selectedModel.value) || 1
      );
      const hazeUrl = changeUrl(uploadRes.url);
      const predRes = await ModelAPI.prediction({
        modelId: Number(selectedModel.value) || 1,
        url: hazeUrl,
        modelParam: dehazeParams.value,
      });
      task.resultUrl = changeUrl(predRes.predUrl);
      task.progress = 100;
      task.status = "success";
    } catch (e) {
      task.status = "failed";
      task.errorMsg = String(e);
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
async function handleRetryBatch(task: BatchTask) {
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
  imageShowStore.setImageUrl(task.resultUrl, ImageTypeEnum.PRED);
  activePage("overlap");
}

// 清空批量任务列表
function handleClearBatch() {
  batchTasks.value = [];
}

function statusTagType(
  status: BatchTask["status"]
): "primary" | "success" | "warning" | "info" | "danger" {
  const map: Record<
    BatchTask["status"],
    "primary" | "success" | "warning" | "info" | "danger"
  > = {
    pending: "info",
    processing: "warning",
    success: "success",
    failed: "danger",
    cancelled: "info",
  };
  return map[status];
}

function statusText(status: BatchTask["status"]) {
  const map: Record<BatchTask["status"], string> = {
    pending: "等待中",
    processing: "处理中",
    success: "已完成",
    failed: "已失败",
    cancelled: "已取消",
  };
  return map[status];
}

function handleExampleImageClick(url: string) {
  imageShowStore.setLoading(true);
  imageShowStore.setImageUrl(url, ImageTypeEnum.HAZE);
  cleanUrl.value = exampleImages.value.filter(
    (item) => item.haze === url
  )[0].clean;
  activePage("singleImage");
  imageShowStore.setLoading(false);
}

// 获取模型选项列表
const getAlgorithmList = async () => {
  await algorithmStore.getAlgorithmOptions();
  modelOptions.value = algorithmStore.algorithmOptions;
};

function handleDatasetImageSelect(haze: string, clear: string) {
  imageShowStore.setImageUrl(haze, ImageTypeEnum.HAZE);
  cleanUrl.value = clear;
  dialogVisible.value = false;
  activePage("singleImage");
}

async function handleCleanUrl(url: string) {
  const res = await fetch(url);
  const blob = await res.blob();
  const cleanFile = new File([blob], "clean.jpg", { type: "image/jpeg" });
  const cleanRes = await FileAPI.upload(
    cleanFile,
    Number(selectedModel.value) || 1
  );
  return cleanRes.url;
}

const router = useRouter();
const route = useRoute();

function handleEval() {
  router.push("/evaluation/index").then(async () => {
    imageShowStore.setModelId(Number(selectedModel.value) || 1);
    imageShowStore.setImageUrls(imgUrls.value);
    if (cleanUrl.value !== "") {
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
    activePage("singleImage");
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
    activePage("example");
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
      @on-take-photo="activePage('camera')"
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
            :data="modelOptions"
            placeholder="请选择去雾模型算法"
            style="width: 240px"
            @change="handleSelectModel"
          />
        </div>
        <div class="batch-entry-wrap">
          <el-button type="primary" plain @click="batchDialogVisible = true">
            批量处理
          </el-button>
          <el-button v-if="batchTasks.length > 0" @click="activePage('batch')">
            查看任务列表（{{ batchTasks.length }}）
          </el-button>
        </div>
      </template>
    </AlgorithmToolBar>
    <!-- 右侧功能栏 -->
    <el-card class="flex-center">
      <!-- 样例图片显示 -->
      <ExampleImageSelect
        v-if="show.example"
        :urls="exampleHazeUrls"
        class="example"
        @on-example-select="handleExampleImageClick"
      />
      <!-- 拍照上传 -->
      <Camera
        v-if="show.camera"
        class="camera"
        @on-cancel="activePage('example')"
        @on-save="handleCameraSave"
      />
      <!-- 单图展示 -->
      <SingleImageShow
        v-if="show.singleImage"
        :src="imgUrls[0].url || ''"
        class="single-image"
      />
      <!-- 处理进度显示（5阶段） -->
      <div v-if="show.loading" class="progress-wrap">
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
      <OverlapImageShow v-if="show.overlap" class="overlap" />
      <!-- 处理完成后保存结果 -->
      <div v-if="show.overlap" class="save-result-wrap">
        <el-button :loading="saving" type="success" @click="handleSaveResult">
          保存结果
        </el-button>
        <el-button type="primary" @click="handleEval"> 评估结果 </el-button>
      </div>
      <!-- 批量处理任务列表 -->
      <div v-if="show.batch" class="batch-wrap">
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
              <el-tag :type="statusTagType(scope.row.status)">
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
                @click="handleViewBatchResult(scope.row)"
              >
                查看
              </el-button>
              <el-button
                v-if="scope.row.status === 'failed'"
                link
                type="warning"
                @click="handleRetryBatch(scope.row)"
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
                @click="handleRemoveBatch(scope.row)"
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

  .effect-wrap {
    width: 60vw;
  }

  .ev-all-wrap {
    display: flex;
    justify-content: space-between;
    width: 60vw;
    margin-bottom: 20px;

    .ev-wrap {
      width: 30%;
      min-width: 250px;
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

    .ev-all-wrap {
      display: flex;
      flex-direction: column;
      width: 82vw;
      margin: 0 auto;

      .ev-wrap {
        width: 100%;
        margin: 10px 0;
      }
    }
  }
}
</style>
