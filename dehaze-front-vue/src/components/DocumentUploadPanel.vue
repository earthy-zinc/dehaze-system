<!-- 文档上传面板：单文件/批量/URL 导入/自定义文本，文件上传经分块预览确认后入库 -->
<script lang="ts" setup>
import type { ChunkingStrategy } from "dehaze-sdk-js";
import { AiKnowledgeBaseAPI, FileAPI } from "dehaze-sdk-js";
import type { UploadFile, UploadUserFile } from "element-plus";
import { ElMessage } from "element-plus";
import { ref } from "vue";

defineOptions({ name: "DocumentUploadPanel" });

interface ChunkConfig {
  chunkingStrategy: ChunkingStrategy;
  chunkSize: number;
  chunkOverlap: number;
}

const props = defineProps<{
  knowledgeBaseId: number;
  readonly?: boolean;
  chunkConfig?: ChunkConfig;
}>();

const emit = defineEmits<{
  (e: "uploaded"): void;
}>();

const activeTab = ref("file");
const skipPreview = ref(false);
const fileList = ref<UploadUserFile[]>([]);
const urlForm = ref({ url: "", title: "" });
const textForm = ref({ title: "", content: "" });
const submitting = ref(false);

const previewVisible = ref(false);
const previewFileId = ref<number>();
const pendingFileIds = ref<number[]>([]);

function defaultChunkConfig(): ChunkConfig {
  return (
    props.chunkConfig ?? {
      chunkingStrategy: "fixed",
      chunkSize: 800,
      chunkOverlap: 80,
    }
  );
}

function resetForms() {
  fileList.value = [];
  urlForm.value = { url: "", title: "" };
  textForm.value = { title: "", content: "" };
}

function notifyUploaded() {
  ElMessage.success("文档已提交入库");
  resetForms();
  emit("uploaded");
}

function handleFileChange(_file: UploadFile, files: UploadFile[]) {
  fileList.value = files;
}

async function handleUpload() {
  if (fileList.value.length === 0) {
    ElMessage.warning("请先选择文件");
    return;
  }
  submitting.value = true;
  try {
    const fileIds: number[] = [];
    for (const item of fileList.value) {
      const info = await FileAPI.upload(item.raw as File);
      fileIds.push(info.id);
    }
    if (skipPreview.value) {
      await AiKnowledgeBaseAPI.batchUploadDocuments(props.knowledgeBaseId, {
        fileIds,
      });
      notifyUploaded();
    } else {
      // 预览首文件分块效果，确认后整批入库
      pendingFileIds.value = fileIds;
      previewFileId.value = fileIds[0];
      previewVisible.value = true;
    }
  } catch {
    // 错误已由全局拦截器提示
  } finally {
    submitting.value = false;
  }
}

async function handlePreviewConfirm() {
  submitting.value = true;
  try {
    if (pendingFileIds.value.length === 1) {
      await AiKnowledgeBaseAPI.uploadDocument(props.knowledgeBaseId, {
        fileId: pendingFileIds.value[0],
      });
    } else {
      await AiKnowledgeBaseAPI.batchUploadDocuments(props.knowledgeBaseId, {
        fileIds: pendingFileIds.value,
      });
    }
    notifyUploaded();
  } catch {
    // 错误已由全局拦截器提示，文件列表保留供重试
  } finally {
    submitting.value = false;
  }
}

async function handleImportUrl() {
  if (!urlForm.value.url) {
    ElMessage.warning("请输入网页地址");
    return;
  }
  submitting.value = true;
  try {
    await AiKnowledgeBaseAPI.importUrlDocument(props.knowledgeBaseId, {
      url: urlForm.value.url,
      title: urlForm.value.title || undefined,
    });
    notifyUploaded();
  } catch {
    // 错误已由全局拦截器提示
  } finally {
    submitting.value = false;
  }
}

async function handleCreateText() {
  if (!textForm.value.title || !textForm.value.content) {
    ElMessage.warning("请填写标题和内容");
    return;
  }
  submitting.value = true;
  try {
    await AiKnowledgeBaseAPI.createTextDocument(props.knowledgeBaseId, {
      title: textForm.value.title,
      content: textForm.value.content,
    });
    notifyUploaded();
  } catch {
    // 错误已由全局拦截器提示
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div v-if="!readonly">
    <el-tabs v-model="activeTab">
      <el-tab-pane label="文件上传" name="file">
        <el-upload
          v-model:file-list="fileList"
          :auto-upload="false"
          multiple
          drag
          class="w-full"
        >
          <div class="py-4">
            <div>将文件拖到此处，或点击选择（支持多选）</div>
            <div class="mt-1 text-xs text-gray-400">
              上传后先预览分块效果，确认后入库
            </div>
          </div>
        </el-upload>
        <div class="mt-3 flex items-center justify-between">
          <el-checkbox v-model="skipPreview">
            跳过分块预览，按库默认配置直接入库
          </el-checkbox>
          <el-button type="primary" :loading="submitting" @click="handleUpload">
            上传并入库
          </el-button>
        </div>
      </el-tab-pane>

      <el-tab-pane label="URL 导入" name="url">
        <el-form label-width="80px">
          <el-form-item label="网页地址">
            <el-input v-model="urlForm.url" placeholder="https://..." />
          </el-form-item>
          <el-form-item label="文档标题">
            <el-input v-model="urlForm.title" placeholder="默认取网页标题" />
          </el-form-item>
          <el-form-item>
            <el-button
              type="primary"
              :loading="submitting"
              @click="handleImportUrl"
            >
              导入
            </el-button>
          </el-form-item>
        </el-form>
      </el-tab-pane>

      <el-tab-pane label="自定义文本" name="text">
        <el-form label-width="80px">
          <el-form-item label="标题">
            <el-input v-model="textForm.title" placeholder="请输入文档标题" />
          </el-form-item>
          <el-form-item label="内容">
            <el-input
              v-model="textForm.content"
              type="textarea"
              :rows="8"
              placeholder="请输入文档内容"
            />
          </el-form-item>
          <el-form-item>
            <el-button
              type="primary"
              :loading="submitting"
              @click="handleCreateText"
            >
              创建文档
            </el-button>
          </el-form-item>
        </el-form>
      </el-tab-pane>
    </el-tabs>

    <ChunkPreviewDialog
      v-model:visible="previewVisible"
      :file-id="previewFileId"
      :chunk-config="defaultChunkConfig()"
      @confirm="handlePreviewConfirm"
    />
  </div>
</template>
