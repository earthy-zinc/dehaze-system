<script lang="ts" setup>
import { ImportModule, ImportResult } from "dehaze-sdk-js";
import { UploadFilled, Download } from "@element-plus/icons-vue";
import type { UploadFile, UploadInstance, UploadRawFile } from "element-plus";
import { genFileId } from "element-plus";
import { useImportExport } from "@/composables/useImportExport";
import ImportResultPanel from "./ImportResultPanel.vue";

defineOptions({
  name: "ImportDialog",
});

const props = defineProps<{
  modelValue: boolean;
  module: ImportModule;
  extraImportParams?: Record<string, unknown>;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "import-complete"): void;
  (e: "async-task-created", taskId: string): void;
}>();

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit("update:modelValue", val),
});

const moduleRef = computed(() => props.module);
const queryParamsRef = ref<Record<string, any>>({});
const extraImportParamsRef = computed(() => props.extraImportParams);

const { importLoading, downloadTemplate, importData } = useImportExport({
  module: moduleRef,
  queryParams: queryParamsRef,
  extraImportParams: extraImportParamsRef,
});

const uploadRef = ref<UploadInstance>();
const selectedFile = ref<File>();
const importMode = ref<"all" | "partial">("all");
const syncResult = ref<ImportResult | null>(null);
const asyncTaskId = ref<string | null>(null);

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ACCEPT_EXTENSIONS = [".xlsx", ".xls", ".csv"];

const dialogTitle = computed(() => `导入${moduleLabel(props.module)}`);

function moduleLabel(m: ImportModule): string {
  const map: Record<ImportModule, string> = {
    user: "用户",
    role: "角色",
    dept: "部门",
    menu: "菜单",
    dict: "字典",
    algorithm: "算法",
  };
  return map[m] ?? m;
}

function resetState() {
  selectedFile.value = undefined;
  importMode.value = "all";
  syncResult.value = null;
  asyncTaskId.value = null;
  uploadRef.value?.clearFiles();
}

function handleClose() {
  resetState();
  visible.value = false;
}

function validateFile(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPT_EXTENSIONS.includes(ext)) {
    return "仅支持 .xlsx、.xls、.csv 格式文件";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "文件大小不能超过 20MB";
  }
  return null;
}

function handleFileChange(uploadFile: UploadFile) {
  if (!uploadFile.raw) return;
  const error = validateFile(uploadFile.raw);
  if (error) {
    ElMessage.warning(error);
    uploadRef.value?.clearFiles();
    return;
  }
  selectedFile.value = uploadFile.raw;
  syncResult.value = null;
  asyncTaskId.value = null;
}

function handleFileExceed(files: File[]) {
  if (files.length === 0) return;
  uploadRef.value?.clearFiles();
  const file = files[0] as UploadRawFile;
  file.uid = genFileId();
  uploadRef.value?.handleStart(file);
  const error = validateFile(file);
  if (error) {
    ElMessage.warning(error);
    return;
  }
  selectedFile.value = file;
}

async function handleDownloadTemplate(format: "excel" | "csv") {
  await downloadTemplate(format);
}

async function handleSubmit() {
  if (!selectedFile.value) {
    ElMessage.warning("请先选择文件");
    return;
  }
  try {
    const result = await importData(selectedFile.value, importMode.value);
    if (result.isAsync) {
      asyncTaskId.value = result.taskId;
      ElMessage.success("数据量较大，已创建导入任务，可在任务列表查看进度");
      emit("async-task-created", result.taskId);
      emit("import-complete");
      handleClose();
    } else {
      syncResult.value = result.result;
      ElMessage.success("导入完成");
      emit("import-complete");
    }
  } catch (e: any) {
    ElMessage.error(e.message || "导入失败");
  }
}

const canSubmit = computed(() => !!selectedFile.value && !importLoading.value);
</script>

<template>
  <el-dialog
    v-model="visible"
    :title="dialogTitle"
    width="640px"
    :close-on-click-modal="false"
    append-to-body
    @close="handleClose"
  >
    <div v-if="!syncResult && !asyncTaskId" class="import-form">
      <el-form label-width="100px">
        <el-form-item label="导入模式">
          <el-radio-group v-model="importMode">
            <el-radio value="all">全量导入</el-radio>
            <el-radio value="partial">部分导入</el-radio>
          </el-radio-group>
          <div class="mode-tip">
            {{
              importMode === "all"
                ? "全量导入：覆盖更新已存在的记录，新增不存在的记录"
                : "部分导入：仅新增不存在的记录，已存在的记录跳过"
            }}
          </div>
        </el-form-item>

        <el-form-item label="下载模板">
          <el-button
            plain
            size="small"
            @click="handleDownloadTemplate('excel')"
          >
            <el-icon><Download /></el-icon>
            Excel 模板
          </el-button>
          <el-button
            plain
            size="small"
            class="ml-2"
            @click="handleDownloadTemplate('csv')"
          >
            <el-icon><Download /></el-icon>
            CSV 模板
          </el-button>
        </el-form-item>

        <el-form-item label="数据文件">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :limit="1"
            :on-change="handleFileChange"
            :on-exceed="handleFileExceed"
            :show-file-list="true"
            accept=".xlsx,.xls,.csv"
            action="#"
            drag
            class="upload-area"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              将文件拖到此处，或<em>点击上传</em>
            </div>
            <template #tip>
              <div class="upload-tip">
                支持 .xlsx、.xls、.csv 格式，文件大小 ≤ 20MB
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
    </div>

    <ImportResultPanel v-else-if="syncResult" :result="syncResult" />

    <el-result
      v-else-if="asyncTaskId"
      icon="info"
      title="已创建导入任务"
      sub-title="数据量较大，已转为异步任务，可在任务列表查看进度"
    />

    <template #footer>
      <el-button @click="handleClose">
        {{ syncResult || asyncTaskId ? "关闭" : "取消" }}
      </el-button>
      <el-button
        v-if="!syncResult && !asyncTaskId"
        type="primary"
        :loading="importLoading"
        :disabled="!canSubmit"
        @click="handleSubmit"
      >
        确定导入
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.import-form {
  .mode-tip {
    margin-top: 4px;
    font-size: 12px;
    color: var(--el-text-color-secondary);
  }
}

.upload-area {
  width: 100%;
}

.upload-tip {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
