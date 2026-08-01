<!-- 文件列表 -->
<script lang="ts" setup>
import { FileAPI, FileInfo, FileQuery } from "dehaze-sdk-js";
import { ElMessage, ElMessageBox, ElTable } from "element-plus";
import { computed, onMounted, reactive, ref } from "vue";
import {
  Document,
  Download,
  Delete,
  Plus,
  Search,
  Refresh,
  UploadFilled,
} from "@element-plus/icons-vue";

defineOptions({
  name: "FileList",
  inheritAttrs: false,
});

const loading = ref(false);
const tableData = ref<FileInfo[]>([]);
const total = ref(0);
const queryFormRef = ref();
const tableRef = ref<any>(null);

const queryParams = reactive<FileQuery>({
  keywords: "",
  pageNum: 1,
  pageSize: 10,
});

const uploadDialog = reactive({
  visible: false,
  fileList: [] as any[],
  uploading: false,
});

function handleQuery() {
  queryParams.pageNum = 1;
  loadData();
}

function resetQuery() {
  queryFormRef.value?.resetFields();
  queryParams.keywords = "";
  handleQuery();
}

async function loadData() {
  loading.value = true;
  try {
    const result = await FileAPI.getPage({
      ...queryParams,
    });
    tableData.value = result.list || [];
    total.value = result.total || 0;
  } catch (e) {
    tableData.value = [];
    total.value = 0;
  } finally {
    loading.value = false;
  }
}

function handleSizeChange(val: number) {
  queryParams.pageSize = val;
  loadData();
}

function handleCurrentChange(val: number) {
  queryParams.pageNum = val;
  loadData();
}

async function handleDelete(row: FileInfo) {
  try {
    await ElMessageBox.confirm(`确认删除文件 "${row.name}" ？`, "删除确认", {
      type: "warning",
      confirmButtonText: "确定",
      cancelButtonText: "取消",
    });
    await FileAPI.deleteById(row.id);
    ElMessage.success("删除文件成功");
    loadData();
  } catch (e) {
    if (e !== "cancel") {
      // 用户取消外其他错误已由全局拦截器提示
    }
  }
}

async function handleDownload(row: FileInfo) {
  if (!row.objectName) {
    ElMessage.warning("文件对象名缺失，无法下载");
    return;
  }
  try {
    const blob = await FileAPI.download(row.objectName);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = row.name || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    // 错误已由全局拦截器提示
  }
}

function openUploadDialog() {
  uploadDialog.visible = true;
  uploadDialog.fileList = [];
}

function handleFileChange(file: any) {
  uploadDialog.fileList = [file];
}

async function submitUpload() {
  if (uploadDialog.fileList.length === 0) {
    ElMessage.warning("请先选择文件");
    return;
  }
  uploadDialog.uploading = true;
  try {
    const rawFile = uploadDialog.fileList[0].raw;
    await FileAPI.upload(rawFile);
    ElMessage.success("文件上传成功");
    uploadDialog.visible = false;
    loadData();
  } catch (e) {
    // 错误已由全局拦截器提示
  } finally {
    uploadDialog.uploading = false;
  }
}

const hasSelection = computed(() => {
  return tableRef.value?.getSelectionRows()?.length > 0;
});

onMounted(() => {
  loadData();
});
</script>

<template>
  <div class="app-container">
    <!-- 搜索区域 -->
    <div class="search-container">
      <el-form ref="queryFormRef" :inline="true" :model="queryParams">
        <el-form-item label="关键字" prop="keywords">
          <el-input
            v-model="queryParams.keywords"
            clearable
            placeholder="文件名/文件类型"
            style="width: 200px"
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
    </div>

    <el-card shadow="never" class="!border-none">
      <!-- 工具栏 -->
      <div class="flex justify-between mb-4">
        <div>
          <el-button type="success" @click="openUploadDialog">
            <el-icon><Plus /></el-icon>
            上传文件
          </el-button>
        </div>
      </div>

      <!-- 文件表格 -->
      <el-table
        ref="tableRef"
        v-loading="loading"
        :data="tableData as FileInfo[]"
        border
        row-key="id"
      >
        <el-table-column type="selection" width="44" align="center" />
        <el-table-column
          label="文件名"
          prop="name"
          min-width="220"
          show-overflow-tooltip
        >
          <template #default="{ row }">
            <div class="file-name-cell">
              <el-icon class="file-icon"><Document /></el-icon>
              <span class="file-name-text">{{ (row as FileInfo).name }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column
          label="文件类型"
          prop="type"
          width="110"
          align="center"
        >
          <template #default="{ row }">
            <el-tag v-if="(row as FileInfo).type" type="info" size="small">{{
              (row as FileInfo).type
            }}</el-tag>
            <span v-else class="text-muted">-</span>
          </template>
        </el-table-column>
        <el-table-column label="文件大小" prop="size" width="120" align="right">
          <template #default="{ row }">
            <span v-if="(row as FileInfo).size" class="file-size">{{
              (row as FileInfo).size
            }}</span>
            <span v-else class="text-muted">-</span>
          </template>
        </el-table-column>
        <el-table-column
          label="创建时间"
          prop="createTime"
          width="180"
          align="center"
        />
        <el-table-column label="操作" width="180" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              type="primary"
              link
              size="small"
              @click="handleDownload(row as FileInfo)"
            >
              <el-icon><Download /></el-icon>
              下载
            </el-button>
            <el-button
              type="danger"
              link
              size="small"
              @click="handleDelete(row as FileInfo)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="queryParams.pageNum"
          v-model:page-size="queryParams.pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          background
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 上传弹窗 -->
    <el-dialog
      v-model="uploadDialog.visible"
      title="上传文件"
      width="500px"
      append-to-body
    >
      <el-upload
        :auto-upload="false"
        :limit="1"
        :on-change="handleFileChange"
        :file-list="uploadDialog.fileList"
        drag
        accept="*/*"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
      </el-upload>

      <template #footer>
        <el-button @click="uploadDialog.visible = false">取消</el-button>
        <el-button
          type="primary"
          :loading="uploadDialog.uploading"
          @click="submitUpload"
        >
          确定上传
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style lang="scss" scoped>
.search-container {
  padding: 16px;
  margin-bottom: 16px;
  background: #fff;
  border-radius: 4px;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.file-name-cell {
  display: flex;
  gap: 8px;
  align-items: center;

  .file-icon {
    flex-shrink: 0;
    font-size: 16px;
    color: var(--el-color-primary);
  }

  .file-name-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
}

.file-size {
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.text-muted {
  color: var(--el-text-color-placeholder);
}
</style>
