<script lang="ts" setup>
import AlgorithmToolBar from "@/components/AlgorithmToolBar/index.vue";
import Camera from "@/components/Camera/index.vue";
import DatasetImageSelect from "@/components/DatasetImageSelect/index.vue";
import ExampleImageSelect from "@/components/ExampleImageSelect/index.vue";
import OverlapImageShow from "@/components/OverlapImageShow/index.vue";
import SingleImageShow from "@/components/SingleImageShow/index.vue";
import { ImageTypeEnum } from "@/enums/ImageTypeEnum";
import { useImageShowStore } from "@/store/modules/imageShow";
import examples from "@/views/presentation/dehaze/exampleImages";
import { groupModelsByTaskTag } from "@/views/presentation/dehaze/modelGroups";
import {
  AiBillingAPI,
  AiImageAPI,
  AiModelAPI,
  AiModelVO,
  FileAPI,
  ImageInputHistoryAPI,
} from "dehaze-sdk-js";
import { UploadFile, UploadUserFile } from "element-plus";

defineOptions({ name: "PresentationDehaze" });
const imageShowStore = useImageShowStore();

const { imageInfo } = toRefs(imageShowStore);
const { images } = toRefs(imageInfo.value);
const { urls: imgUrls } = toRefs(images.value);

const exampleHazeUrls = computed(() => examples.map((item) => item.haze));
const cleanUrl = ref("");
// 图像模型标识（统一模型规范 slug），提交任务与评估/对比页共用
const selectedModel = ref<string>();
const modelOptions = ref<AiModelVO[]>([]);

// 分组规则与跨端口径见 modelGroups（与 flutter 同口径，其余端为扁平列表）
const groupedModelOptions = computed(() =>
  groupModelsByTaskTag(modelOptions.value)
);

// 当前激活的页面（单值状态机，替代多个布尔标志）
type PageName =
  "camera" | "singleImage" | "example" | "overlap" | "loading" | "batch";
const activePage = ref<PageName>("example");

// 最近一次图像来源（处理成功后回写云端历史用）
type InputSource = "upload" | "camera" | "sample";
let lastInputSource: InputSource | undefined;

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
  handleImageUpload(file, "camera");
}

function handleSelectModel(modelId: string) {
  imageShowStore.modelId = modelId;
}

function handleImageUpload(file: File, source: InputSource = "upload") {
  lastInputSource = source;
  imageShowStore.loading = true;
  FileAPI.upload(file)
    .then((res) => {
      // 文件上传成功后拿到服务器返回的 url 地址在右侧渲染
      imageShowStore.setImageUrl(res.url, ImageTypeEnum.HAZE);
      activePage.value = "singleImage";
    })
    .catch(() => {})
    .finally(() => {
      imageShowStore.loading = false;
    });
}

function handleReset() {
  imageShowStore.imageInfo.images.urls = [];
  imageShowStore.magnifierInfo.enabled = false;
  cleanUrl.value = "";
  activePage.value = "example";
}

// 在途图像任务号（提交/轮询回调回填，供取消使用）
let currentTaskId: number | undefined;

/** 取消后端在途任务（幂等；任务已终态或请求中断时忽略，不阻断交互） */
async function cancelRunningTask() {
  if (!currentTaskId) return;
  try {
    await AiImageAPI.cancelImageTask(currentTaskId);
  } catch {}
}

// 取消处理
async function handleCancelProcess() {
  cancelFlag = true;
  processing.value = false;
  stopProgressSimulation();
  progress.value = 0;
  activePage.value = "singleImage";
  await cancelRunningTask();
  ElMessage.info("已取消处理");
}

// 当前重试倒计时
const retryCountdown = ref(0);
const showRetryPanel = ref(false);
// 失败原因：优先用后端返回的 errorMessage（引擎/模型类错误不能归因成网络问题）
const failureMessage = ref("");
let retryTimer: number | undefined;

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
  // 积分余额预提示（真实扣减与校验由服务端按张执行）
  try {
    const balance = await AiBillingAPI.getBalance();
    // creditsBalance 为后端 Decimal 字符串，比较前需 Number() 转换
    if (Number(balance.creditsBalance) <= 0) {
      const confirmed = await ElMessageBox.confirm(
        `当前积分余额为 ${balance.creditsBalance}，图像处理按张扣减积分，请及时充值。`,
        "积分不足",
        {
          confirmButtonText: "去充值",
          cancelButtonText: "取消",
          type: "warning",
        }
      );
      if (confirmed) {
        ElMessage.info("请联系管理员充值");
      }
      return;
    }
  } catch (e: any) {
    ElMessage.warning("无法获取积分余额：" + (e.message || "未知错误"));
  }
  // 显示确认对话框
  const modelName =
    modelOptions.value.find((m) => m.modelId === selectedModel.value)
      ?.displayName ?? selectedModel.value;
  const imgUrl = imgUrls.value[0].url;
  const imageName = imgUrl.split("/").pop() || "未命名图片";
  try {
    await ElMessageBox.confirm(
      [
        `图片：${imageName}`,
        `模型：${modelName}`,
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
  showRetryPanel.value = false;
  failureMessage.value = "";
  retryCountdown.value = 0;
  const modelId = selectedModel.value;
  cancelFlag = false;
  processing.value = true;
  progress.value = 0;
  imageShowStore.loading = true;
  imageShowStore.modelId = modelId;
  activePage.value = "loading";
  // 启动模拟进度
  startProgressSimulation(95);
  AiImageAPI.submitAndWait(
    {
      model: modelId,
      imageUrl: imgUrl,
      params: dehazeParams.value
        ? JSON.stringify(dehazeParams.value)
        : undefined,
    },
    {
      onPoll: (_status, taskId) => {
        currentTaskId = taskId ?? currentTaskId;
      },
    }
  )
    .then(async (res) => {
      if (cancelFlag) {
        await cancelRunningTask();
        return;
      }
      if (res.status === 3) {
        throw new Error(res.errorMessage || "去雾处理失败");
      }
      progress.value = 95;
      stopProgressSimulation();
      imageShowStore.setImageUrl(imgUrl, ImageTypeEnum.HAZE);
      imageShowStore.setImageUrl(res.resultUrl || "", ImageTypeEnum.PRED);
      imageShowStore.predLogId = res.taskId || 0;
      // 回写云端历史（F-M04-007），失败不阻断处理主流程
      ImageInputHistoryAPI.create({
        originalImageUrl: imgUrl,
        resultImageUrl: res.resultUrl,
        modelId,
        modelName,
        modelParams: dehazeParams.value
          ? JSON.stringify(dehazeParams.value)
          : undefined,
        // res.time 为秒，输入历史 processing_time 列为毫秒
        processingTime: res.time != null ? res.time * 1000 : undefined,
        status: 1,
        inputSource: lastInputSource,
      }).catch(() => {});
      if (cleanUrl.value) {
        try {
          cleanUrl.value = await handleCleanUrl(cleanUrl.value);
        } catch {}
      }
      progress.value = 100;
      activePage.value = "overlap";
    })
    .catch((err) => {
      if (cancelFlag) return;
      failureMessage.value =
        err instanceof Error && err.message
          ? err.message
          : "处理失败，请稍后重试";
      showRetryPanel.value = true;
      retryCountdown.value = 3;
      startRetryCountdown();
      activePage.value = "singleImage";
    })
    .finally(() => {
      stopProgressSimulation();
      processing.value = false;
      imageShowStore.loading = false;
    });
}

// 重试处理
function startRetryCountdown() {
  stopRetryCountdown();
  retryTimer = window.setInterval(() => {
    retryCountdown.value--;
    if (retryCountdown.value <= 0) {
      stopRetryCountdown();
    }
  }, 1000);
}

function stopRetryCountdown() {
  if (retryTimer !== undefined) {
    clearInterval(retryTimer);
    retryTimer = undefined;
  }
}

function handleRetry() {
  showRetryPanel.value = false;
  retryCountdown.value = 0;
  stopRetryCountdown();
  handleGenerateImage();
}

onUnmounted(() => {
  stopProgressSimulation();
  stopRetryCountdown();
  if (batchTimer !== undefined) {
    clearInterval(batchTimer);
  }
});

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
    await FileAPI.upload(file);
    ElMessage.success("结果保存成功");
  } catch (e: any) {
    // axios 类错误（HTTP/业务码）由全局钩子提示，这里只兜底原生 fetch 抛出的本地错误
    if (!e.isAxiosError) {
      ElMessage.error("保存失败：" + (e.message || "未知错误"));
    }
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
  taskId: number;
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
      taskId: 0,
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
  imageShowStore.modelId = modelId;
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
      const uploadRes = await FileAPI.upload(task.file);
      task.hazeUrl = uploadRes.url;
      const predRes = await AiImageAPI.submitAndWait({
        model: modelId,
        imageUrl: task.hazeUrl,
        params: dehazeParams.value
          ? JSON.stringify(dehazeParams.value)
          : undefined,
      });
      if (predRes.status === 3) {
        throw new Error(predRes.errorMessage || "处理失败");
      }
      task.resultUrl = predRes.resultUrl || "";
      task.taskId = predRes.taskId || 0;
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
  imageShowStore.imageInfo.images.urls = [];
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

const statusOptions: { label: string; value: string; color: string }[] = [
  { label: "等待中", value: "pending", color: "#1890ff" },
  { label: "处理中", value: "processing", color: "#1890ff" },
  { label: "已完成", value: "success", color: "#52c41a" },
  { label: "已失败", value: "failed", color: "#ff4d4f" },
  { label: "已取消", value: "cancelled", color: "#8c8c8c" },
];

function statusTagColor(status: string): string {
  return statusOptions.find((o) => o.value === status)!.color;
}

function statusText(status: string) {
  return statusOptions.find((o) => o.value === status)!.label;
}

function handleExampleImageClick(url: string) {
  const matched = examples.find((item) => item.haze === url);
  if (!matched) return;
  lastInputSource = "sample";
  imageShowStore.setImageUrl(url, ImageTypeEnum.HAZE);
  cleanUrl.value = matched.clean;
  activePage.value = "singleImage";
}

// 加载可选图像模型：默认选中最近一次使用且仍启用的模型（既定预选规则），
// 无历史记录时取后端标记的平台默认模型（isDefault ← 字典 ai_model.default_model.image_edit）；
// 后端未配置平台默认时不下选，由用户显式选择（不再回落到"启用列表首项"猜值）
async function loadModels() {
  modelOptions.value = await AiModelAPI.listEnabledModels("image_edit");
  let historyModelId: string | undefined;
  try {
    const history = await ImageInputHistoryAPI.getPage({
      pageNum: 1,
      pageSize: 1,
    });
    historyModelId = history.list[0]?.modelId;
  } catch {}
  const matched = modelOptions.value.find((m) => m.modelId === historyModelId);
  const preferred = matched ?? modelOptions.value.find((m) => m.isDefault);
  if (preferred && !selectedModel.value) {
    selectedModel.value = preferred.modelId;
    imageShowStore.modelId = preferred.modelId;
  }
}

function handleDatasetImageSelect(haze: string, clear: string) {
  lastInputSource = undefined;
  imageShowStore.setImageUrl(haze, ImageTypeEnum.HAZE);
  cleanUrl.value = clear;
  dialogVisible.value = false;
  activePage.value = "singleImage";
}

async function handleCleanUrl(url: string) {
  try {
    const res = await fetch(url);
    const blob = await res.blob();
    const cleanFile = new File([blob], "clean.jpg", { type: "image/jpeg" });
    const cleanRes = await FileAPI.upload(cleanFile);
    return cleanRes.url;
  } catch (e: any) {
    throw new Error("清晰图处理失败：" + (e.message || "未知错误"));
  }
}

const router = useRouter();
const route = useRoute();

function handleEval() {
  if (!selectedModel.value) {
    ElMessage.error("请先选择去雾模型");
    return;
  }
  router
    .push({
      path: "/evaluation/index",
      query: { params: JSON.stringify(dehazeParams.value) },
    })
    .then(() => {
      imageShowStore.modelId = selectedModel.value!;
      imageShowStore.imageInfo.images.urls = imgUrls.value;
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
    imageShowStore.imageInfo.images.urls = [];
    imageShowStore.setImageUrl(imageUrl, ImageTypeEnum.HAZE);
    activePage.value = "singleImage";
    // 清除 query 参数，避免回退时重复加载
    router.replace({ path: "/presentation/dehaze" });
    return true;
  }
  return false;
}

onMounted(() => {
  imageShowStore.loading = true;
  loadModels();
  imageShowStore.imageInfo.images.urls = [];
  if (!loadImageFromQuery()) {
    activePage.value = "example";
  }
  imageShowStore.loading = false;
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
          <el-select
            v-model="selectedModel"
            data-testid="model-select"
            placeholder="请选择去雾模型"
            style="width: 240px"
            @change="handleSelectModel"
          >
            <el-option-group
              v-for="group in groupedModelOptions"
              :key="group.label"
              :label="group.label"
            >
              <el-option
                v-for="model in group.models"
                :key="model.modelId"
                :label="model.displayName"
                :value="model.modelId"
              />
            </el-option-group>
          </el-select>
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

      <!-- 失败重试面板 -->
      <div
        v-if="showRetryPanel && activePage === 'singleImage'"
        class="retry-panel"
      >
        <el-alert
          :title="failureMessage"
          type="error"
          :closable="false"
          show-icon
        />
        <div class="retry-actions">
          <el-button
            :disabled="retryCountdown > 0"
            type="primary"
            @click="handleRetry"
          >
            {{
              retryCountdown > 0 ? `自动重试(${retryCountdown})` : "手动重试"
            }}
          </el-button>
          <el-button @click="showRetryPanel = false">关闭</el-button>
        </div>
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

  .retry-panel {
    width: calc(64vw - 6vw);
    padding: 16px;
    margin: 16px auto;
    background: #fff5f5;
    border: 1px solid #fde2e2;
    border-radius: 8px;

    .retry-actions {
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-top: 12px;
    }
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
