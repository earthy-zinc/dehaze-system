<script lang="ts" setup>
import Camera from "@/components/Camera/index.vue";
import ExampleImageSelect from "@/components/ExampleImageSelect/index.vue";
import { useImageShowStore } from "@/store/modules/imageShow";
import examples from "@/views/presentation/dehaze/exampleImages";
import { FileAPI } from "dehaze-sdk-js";
import { UploadFilled } from "@element-plus/icons-vue";
import type { UploadRawFile, UploadRequestOptions } from "element-plus";

defineOptions({
  name: "ImageInput",
});

const router = useRouter();
const imageShowStore = useImageShowStore();

// 历史记录存储 key
const HISTORY_KEY = "dehaze:image-history";
// 支持的文件格式（仅 JPG / PNG）
const ACCEPT_FORMATS = ".jpg,.jpeg,.png";
const ALLOWED_EXTS = [".jpg", ".jpeg", ".png"];
// 最大文件大小：20MB
const MAX_FILE_SIZE = 20 * 1024 * 1024;

// 当前激活的 Tab
const activeTab = ref<"upload" | "camera" | "sample" | "history">("upload");

// ========== 上传面板状态 ==========
const uploading = ref(false);
const uploadProgress = ref(0);
const previewUrl = ref("");
const dragOver = ref(false);

// ========== 样例面板状态 ==========
const sampleUrls = computed(() => examples.map((item) => item.haze));

// ========== 历史记录状态 ==========
interface HistoryRecord {
  id: string;
  url: string;
  thumbnail: string;
  time: number;
  source: "upload" | "camera" | "sample";
}

const sourceLabels: Record<HistoryRecord["source"], string> = {
  upload: "上传",
  camera: "拍照",
  sample: "样例",
};

const historyRecords = ref<HistoryRecord[]>([]);

// 从 localStorage 加载历史记录
function loadHistory() {
  const data = localStorage.getItem(HISTORY_KEY);
  if (data) {
    try {
      historyRecords.value = JSON.parse(data);
    } catch {
      historyRecords.value = [];
    }
  }
}

// 保存历史记录到 localStorage
function saveHistory(records: HistoryRecord[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(records));
}

// 添加历史记录
function addHistoryRecord(url: string, source: "upload" | "camera" | "sample") {
  const record: HistoryRecord = {
    id: Date.now().toString() + Math.random().toString(36).slice(2),
    url,
    thumbnail: url,
    time: Date.now(),
    source,
  };
  historyRecords.value.unshift(record);
  // 最多保留 20 条
  if (historyRecords.value.length > 20) {
    historyRecords.value = historyRecords.value.slice(0, 20);
  }
  saveHistory(historyRecords.value);
}

// 删除历史记录
function deleteHistoryRecord(id: string) {
  historyRecords.value = historyRecords.value.filter((r) => r.id !== id);
  saveHistory(historyRecords.value);
}

// 按时间分组展示历史记录
const groupedHistory = computed(() => {
  const now = new Date();
  const todayStart = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate()
  ).getTime();
  const yesterdayStart = todayStart - 24 * 60 * 60 * 1000;
  const sevenDaysAgoStart = todayStart - 7 * 24 * 60 * 60 * 1000;

  const groups = {
    today: [] as HistoryRecord[],
    yesterday: [] as HistoryRecord[],
    recent7: [] as HistoryRecord[],
    earlier: [] as HistoryRecord[],
  };

  historyRecords.value.forEach((record) => {
    if (record.time >= todayStart) {
      groups.today.push(record);
    } else if (record.time >= yesterdayStart) {
      groups.yesterday.push(record);
    } else if (record.time >= sevenDaysAgoStart) {
      groups.recent7.push(record);
    } else {
      groups.earlier.push(record);
    }
  });

  return [
    { label: "今天", records: groups.today },
    { label: "昨天", records: groups.yesterday },
    { label: "最近7天", records: groups.recent7 },
    { label: "更早", records: groups.earlier },
  ].filter((g) => g.records.length > 0);
});

// 格式化时间
function formatTime(timestamp: number) {
  const date = new Date(timestamp);
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${y}-${m}-${d} ${h}:${min}`;
}

// ========== 图片选择统一处理 ==========
function handleImageSelected(
  url: string,
  source: "upload" | "camera" | "sample"
) {
  addHistoryRecord(url, source);
  // 跳转到去雾展示页，通过 query 参数传递图片 URL
  router.push({
    path: "/presentation/dehaze",
    query: { imageUrl: url },
  });
}

// ========== 上传面板 ==========
// 文件上传前校验：格式与大小
function handleBeforeUpload(file: UploadRawFile): boolean {
  const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXTS.includes(ext)) {
    ElMessage.error(`仅支持 JPG / PNG 格式，当前为 ${ext.toUpperCase()} 格式`);
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    ElMessage.error(`图片大小不能超过 20MB，当前大小为 ${sizeMB}MB`);
    return false;
  }
  return true;
}

// 自定义上传请求，支持进度回调
async function handleUploadRequest(options: UploadRequestOptions) {
  uploading.value = true;
  uploadProgress.value = 0;
  previewUrl.value = URL.createObjectURL(options.file);

  try {
    const res = await FileAPI.upload(
      options.file,
      imageShowStore.modelId,
      (e) => {
        if (e.total) {
          uploadProgress.value = Math.round((e.loaded / e.total) * 100);
        }
      }
    );
    uploadProgress.value = 100;
    ElMessage.success("上传成功");
    const url = res.url;
    handleImageSelected(url, "upload");
  } catch (err: any) {
    ElMessage.error("上传失败：" + (err?.message || "未知错误"));
  } finally {
    uploading.value = false;
    setTimeout(() => {
      previewUrl.value = "";
      uploadProgress.value = 0;
    }, 1000);
  }
}

// ========== 拍照面板 ==========
function handleCameraSave(file: File) {
  uploading.value = true;
  FileAPI.upload(file, imageShowStore.modelId)
    .then((res) => {
      const url = res.url;
      handleImageSelected(url, "camera");
    })
    .catch((err: any) => {
      ElMessage.error("上传失败：" + (err?.message || "未知错误"));
    })
    .finally(() => {
      uploading.value = false;
    });
}

// 拍照取消时切回上传 Tab
function handleCameraCancel() {
  activeTab.value = "upload";
}

// ========== 样例面板 ==========
function handleSampleSelect(url: string) {
  handleImageSelected(url, "sample");
}

// ========== 历史面板 ==========
function handleHistoryReprocess(record: HistoryRecord) {
  router.push({
    path: "/presentation/dehaze",
    query: { imageUrl: record.url },
  });
}

function handleHistoryDelete(record: HistoryRecord) {
  ElMessageBox.confirm("确认删除该历史记录吗？删除后不可恢复。", "提示", {
    confirmButtonText: "确定",
    cancelButtonText: "取消",
    type: "warning",
  })
    .then(() => {
      deleteHistoryRecord(record.id);
      ElMessage.success("删除成功");
    })
    .catch(() => {});
}

onActivated(() => {
  loadHistory();
});
</script>

<template>
  <div class="image-input-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2>图像输入</h2>
      <p>选择图片进行去雾处理</p>
    </div>

    <!-- Tab 切换输入方式 -->
    <el-tabs v-model="activeTab" class="input-tabs">
      <!-- 上传面板 -->
      <el-tab-pane label="上传" name="upload">
        <div class="panel upload-panel">
          <el-upload
            drag
            class="upload-area"
            :class="{ 'is-dragover': dragOver }"
            :accept="ACCEPT_FORMATS"
            :show-file-list="false"
            :auto-upload="true"
            :before-upload="handleBeforeUpload"
            :http-request="handleUploadRequest"
            :disabled="uploading"
            @dragenter="dragOver = true"
            @dragleave="dragOver = false"
            @dragover="dragOver = true"
            @drop="dragOver = false"
          >
            <el-icon class="el-icon--upload">
              <UploadFilled />
            </el-icon>
            <div class="el-upload__text">
              拖拽图片到此处，或<em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                仅支持 JPG / PNG 格式，单文件不超过 20MB
              </div>
            </template>
          </el-upload>

          <!-- 上传进度 -->
          <div v-if="uploading || uploadProgress > 0" class="upload-progress">
            <el-progress
              :percentage="uploadProgress"
              :status="uploadProgress === 100 ? 'success' : ''"
            />
          </div>

          <!-- 图片预览 -->
          <div v-if="previewUrl" class="preview-wrap">
            <img :src="previewUrl" alt="预览" class="preview-img" />
          </div>
        </div>
      </el-tab-pane>

      <!-- 拍照面板 -->
      <el-tab-pane label="拍照" name="camera">
        <div v-loading="uploading" class="panel camera-panel">
          <Camera
            v-if="activeTab === 'camera'"
            @on-save="handleCameraSave"
            @on-cancel="handleCameraCancel"
          />
        </div>
      </el-tab-pane>

      <!-- 样例面板 -->
      <el-tab-pane label="样例" name="sample">
        <div class="panel sample-panel">
          <!-- 样例图片 -->
          <ExampleImageSelect
            v-if="sampleUrls.length > 0"
            :urls="sampleUrls"
            @on-example-select="handleSampleSelect"
          />
          <el-empty v-else description="暂无样例图片" />
        </div>
      </el-tab-pane>

      <!-- 历史面板 -->
      <el-tab-pane label="历史" name="history">
        <div class="panel history-panel">
          <template v-if="groupedHistory.length > 0">
            <div
              v-for="group in groupedHistory"
              :key="group.label"
              class="history-group"
            >
              <div class="group-title">{{ group.label }}</div>
              <div class="history-list">
                <div
                  v-for="record in group.records"
                  :key="record.id"
                  class="history-card"
                >
                  <img
                    :src="record.thumbnail"
                    alt="历史图片"
                    class="history-thumbnail"
                  />
                  <div class="history-info">
                    <div class="history-time">
                      {{ formatTime(record.time) }}
                    </div>
                    <div class="history-source">
                      来源：{{ sourceLabels[record.source] }}
                    </div>
                  </div>
                  <div class="history-actions">
                    <el-button
                      size="small"
                      type="primary"
                      @click="handleHistoryReprocess(record)"
                    >
                      重新处理
                    </el-button>
                    <el-button
                      size="small"
                      type="danger"
                      plain
                      @click="handleHistoryDelete(record)"
                    >
                      删除
                    </el-button>
                  </div>
                </div>
              </div>
            </div>
          </template>
          <el-empty v-else description="暂无历史记录" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<style lang="scss" scoped>
.image-input-container {
  height: calc(100vh - #{$navbar-height} - #{$tags-view-height});
  padding: 20px;
  overflow-y: auto;
}

.page-header {
  margin-bottom: 20px;
  text-align: center;

  h2 {
    margin: 0 0 8px;
    font-size: 24px;
    font-weight: 700;
  }

  p {
    margin: 0;
    font-size: 14px;
    color: var(--el-text-color-secondary);
  }
}

.input-tabs {
  max-width: 1200px;
  margin: 0 auto;
}

.panel {
  min-height: 400px;
  padding: 20px;
}

/* 上传面板 */
.upload-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;

  .upload-area {
    :deep(.el-upload-dragger) {
      width: 500px;
      max-width: 100%;
      padding: 40px 20px;
      border-color: var(--el-border-color);
      transition: all 0.3s ease;
    }

    :deep(.el-upload-dragger.is-dragover) {
      background-color: rgb(var(--el-color-primary-rgb), 0.06);
      border-color: var(--el-color-primary);
      box-shadow: 0 0 20px rgb(var(--el-color-primary-rgb), 0.2);
    }
  }

  .upload-progress {
    width: 500px;
    max-width: 100%;
    margin-top: 20px;
  }

  .preview-wrap {
    margin-top: 20px;
    text-align: center;

    .preview-img {
      max-width: 100%;
      max-height: 400px;
      border-radius: 8px;
      box-shadow: var(--el-box-shadow-light);
    }
  }
}

/* 拍照面板 */
.camera-panel {
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 样例面板 */
.sample-panel {
  text-align: center;
}

/* 历史面板 */
.history-panel {
  .history-group {
    margin-bottom: 24px;
  }

  .group-title {
    padding-left: 8px;
    margin-bottom: 12px;
    font-size: 16px;
    font-weight: 600;
    border-left: 4px solid var(--el-color-primary);
  }

  .history-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 16px;
  }

  .history-card {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 12px;
    background-color: var(--el-bg-color-page);
    border: 1px solid var(--el-border-color-lighter);
    border-radius: 8px;
    transition: all 0.2s ease;

    &:hover {
      border-color: var(--el-color-primary);
      box-shadow: var(--el-box-shadow-light);
    }
  }

  .history-thumbnail {
    flex-shrink: 0;
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 6px;
  }

  .history-info {
    flex: 1;
    min-width: 0;

    .history-time {
      margin-bottom: 4px;
      font-size: 14px;
      font-weight: 500;
    }

    .history-source {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }
  }

  .history-actions {
    display: flex;
    flex-shrink: 0;
    flex-direction: column;
    gap: 8px;
  }
}

@media screen and (width <= 768px) {
  .upload-panel {
    :deep(.el-upload-dragger) {
      width: 100%;
    }

    .upload-progress {
      width: 100%;
    }
  }

  .history-panel {
    .history-list {
      grid-template-columns: 1fr;
    }

    .history-card {
      flex-direction: column;
      align-items: stretch;
      text-align: center;

      .history-thumbnail {
        width: 100%;
        height: 120px;
      }

      .history-actions {
        flex-direction: row;
        justify-content: center;
      }
    }
  }
}
</style>
