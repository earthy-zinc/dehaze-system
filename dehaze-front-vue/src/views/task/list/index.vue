<!-- 任务中心 -->
<template>
  <div class="app-container">
    <!-- 状态筛选 -->
    <el-card class="search-container" shadow="never">
      <div class="flex items-center justify-between flex-wrap gap-3">
        <div class="flex items-center gap-3 flex-wrap">
          <el-radio-group
            v-model="categoryFilter"
            @change="handleCategoryChange"
          >
            <el-radio-button value="">全部类别</el-radio-button>
            <el-radio-button value="import">导入</el-radio-button>
            <el-radio-button value="export">导出</el-radio-button>
          </el-radio-group>
          <el-radio-group v-model="statusFilter" @change="handleStatusChange">
            <el-radio-button value="">全部</el-radio-button>
            <el-radio-button :value="1">待执行</el-radio-button>
            <el-radio-button :value="2">执行中</el-radio-button>
            <el-radio-button :value="3">已完成</el-radio-button>
            <el-radio-button :value="4">失败</el-radio-button>
            <el-radio-button :value="5">已取消</el-radio-button>
          </el-radio-group>
        </div>
        <el-button @click="loadTaskList"
          ><el-icon><Refresh /></el-icon>刷新</el-button
        >
      </div>
    </el-card>

    <!-- 任务列表 -->
    <el-card class="table-container" shadow="never">
      <el-table
        v-loading="taskStore.loading"
        :data="taskStore.taskList"
        border
        empty-text="暂无任务"
      >
        <el-table-column
          label="任务ID"
          prop="taskId"
          width="300"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="130" align="center">
          <template #default="{ row }">
            {{ taskTypeLabel[row.taskType] ?? row.taskType }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :color="statusTagColor[row.status]" effect="dark">
              {{ statusLabel[row.status] }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="220">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="progressStatus(row.status)"
              :stroke-width="16"
              :text-inside="true"
            />
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="180" align="center">
          <template #default="{ row }">{{
            formatTime(row.createdAt)
          }}</template>
        </el-table-column>
        <el-table-column label="完成时间" width="180" align="center">
          <template #default="{ row }">
            {{ formatTime(row.completedAt) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="240" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="handleDetail(row as TaskVO)">
              详情
            </el-button>
            <el-button
              v-if="canCancel(row.status)"
              type="warning"
              link
              @click="handleCancel(row as TaskVO)"
            >
              取消
            </el-button>
            <el-button
              v-if="row.status === 3 && isImportTask(row.taskType)"
              type="primary"
              link
              :loading="downloadLoadingId === row.taskId"
              @click="handleDownload(row as TaskVO)"
            >
              查看结果
            </el-button>
            <el-button
              v-if="row.status === 3 && !isImportTask(row.taskType)"
              type="success"
              link
              :loading="downloadLoadingId === row.taskId"
              @click="handleDownload(row as TaskVO)"
            >
              下载
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <Pagination
        v-model:page="queryParams.pageNum"
        v-model:limit="queryParams.pageSize"
        :total="taskStore.total"
        @pagination="loadTaskList"
      />
    </el-card>

    <!-- 任务详情弹窗 -->
    <el-dialog v-model="detailVisible" title="任务详情" width="600px">
      <template v-if="taskStore.currentTask">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="任务ID">
            {{ taskStore.currentTask.taskId }}
          </el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag
              :color="statusTagColor[taskStore.currentTask.status]"
              effect="dark"
            >
              {{ statusLabel[taskStore.currentTask.status] }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="进度">
            <el-progress
              :percentage="taskStore.currentTask.progress"
              :status="progressStatus(taskStore.currentTask.status)"
              :stroke-width="16"
              :text-inside="true"
            />
          </el-descriptions-item>
          <el-descriptions-item
            v-if="taskStore.currentTask.totalFiles"
            label="文件处理"
          >
            {{ taskStore.currentTask.processedFiles || 0 }} /
            {{ taskStore.currentTask.totalFiles }}
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ formatTime(taskStore.currentTask.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item label="开始时间">
            {{ formatTime(taskStore.currentTask.startedAt) }}
          </el-descriptions-item>
          <el-descriptions-item label="完成时间">
            {{ formatTime(taskStore.currentTask.completedAt) }}
          </el-descriptions-item>
          <el-descriptions-item
            v-if="taskStore.currentTask.expiresAt"
            label="过期时间"
          >
            {{ formatTime(taskStore.currentTask.expiresAt) }}
          </el-descriptions-item>
          <el-descriptions-item
            v-if="taskStore.currentTask.error"
            label="错误信息"
          >
            <span class="text-red-500">{{ taskStore.currentTask.error }}</span>
          </el-descriptions-item>
        </el-descriptions>
      </template>
      <template #footer>
        <el-button @click="detailVisible = false">关闭</el-button>
        <el-button
          v-if="
            taskStore.currentTask && canCancel(taskStore.currentTask.status)
          "
          type="warning"
          :loading="cancelLoading"
          @click="handleCancel(taskStore.currentTask)"
        >
          取消任务
        </el-button>
        <el-button
          v-if="taskStore.currentTask?.status === 3"
          type="success"
          :loading="downloadLoadingId === taskStore.currentTask?.taskId"
          @click="handleDownload(taskStore.currentTask)"
        >
          下载结果
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { TaskVO, TaskQuery, TaskCategory } from "dehaze-sdk-js";
import { Refresh } from "@element-plus/icons-vue";
import { useTaskStore } from "@/store";

defineOptions({
  name: "TaskList",
  inheritAttrs: false,
});

const taskStore = useTaskStore();

// 查询参数
const queryParams = reactive<TaskQuery>({
  pageNum: 1,
  pageSize: 10,
});

// 状态筛选值（空字符串表示全部）
const statusFilter = ref<number | "">("");
// 类别筛选值（空字符串表示全部）
const categoryFilter = ref<"" | TaskCategory>("");

// 详情弹窗
const detailVisible = ref(false);
// 取消操作加载状态
const cancelLoading = ref(false);
// 下载操作加载状态（记录正在下载的任务ID）
const downloadLoadingId = ref<string | null>(null);

// 状态标签颜色映射（按需求规格 5.3 节）
const statusTagColor: Record<number, string> = {
  1: "#1890ff",
  2: "#1890ff",
  3: "#52c41a",
  4: "#ff4d4f",
  5: "#8c8c8c",
};

// 状态标签文本映射
const statusLabel: Record<number, string> = {
  1: "待执行",
  2: "执行中",
  3: "已完成",
  4: "失败",
  5: "已取消",
};

// 任务类型文本映射
const taskTypeLabel: Record<string, string> = {
  dataset_export: "数据集导出",
  item_download: "数据项下载",
  batch_download: "批量下载",
  custom_export: "自定义导出",
  user_export: "用户导出",
  role_export: "角色导出",
  dept_export: "部门导出",
  menu_export: "菜单导出",
  dict_export: "字典导出",
  algorithm_export: "算法导出",
  user_import: "用户导入",
  role_import: "角色导入",
  dept_import: "部门导入",
  menu_import: "菜单导入",
  dict_import: "字典导入",
  algorithm_import: "算法导入",
};

// 判断是否为导入任务
function isImportTask(taskType?: string): boolean {
  if (!taskType) return false;
  return taskType.endsWith("_import");
}

// 需要轮询的任务状态
const POLLING_STATUSES = [1, 2];

/**
 * 格式化时间显示
 */
function formatTime(t?: Date | string): string {
  if (!t) return "-";
  return new Date(t).toLocaleString("zh-CN");
}

/**
 * 获取进度条状态
 * @param status 任务状态
 */
function progressStatus(
  status: number
): "" | "success" | "exception" | "warning" {
  if (status === 3) return "success";
  if (status === 4) return "exception";
  return "";
}

/**
 * 判断任务是否可取消
 * @param status 任务状态
 */
function canCancel(status: number): boolean {
  return POLLING_STATUSES.includes(status);
}

/**
 * 加载任务列表
 */
async function loadTaskList() {
  try {
    await taskStore.getTaskList(queryParams);
  } catch (e: any) {
    ElMessage.error(e.message || "加载任务列表失败");
    return;
  }
  // 存在进行中的任务时启动轮询，否则停止
  const hasActiveTasks = taskStore.taskList.some((t) =>
    POLLING_STATUSES.includes(t.status)
  );
  if (hasActiveTasks) {
    taskStore.startPolling();
  } else {
    taskStore.stopPolling();
  }
}

/**
 * 状态筛选变化
 */
function handleStatusChange() {
  queryParams.status = (statusFilter.value || undefined) as TaskQuery["status"];
  queryParams.pageNum = 1;
  loadTaskList();
}

/**
 * 类别筛选变化
 */
function handleCategoryChange() {
  queryParams.taskCategory = (categoryFilter.value ||
    undefined) as TaskQuery["taskCategory"];
  queryParams.pageNum = 1;
  loadTaskList();
}

/**
 * 查看任务详情
 * @param row 任务行数据
 */
function handleDetail(row: TaskVO) {
  taskStore.currentTask = row;
  detailVisible.value = true;
}

/**
 * 取消任务（二次确认）
 * @param task 任务信息
 */
async function handleCancel(task: TaskVO) {
  try {
    await ElMessageBox.confirm("确认取消该任务吗？", "提示", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    });
  } catch {
    return; // 用户取消确认
  }
  cancelLoading.value = true;
  try {
    await taskStore.cancelTask(task.taskId);
    ElMessage.success("任务已取消");
    // 同步更新当前查看的任务状态
    if (taskStore.currentTask?.taskId === task.taskId) {
      taskStore.currentTask = {
        ...taskStore.currentTask,
        status: 5,
        completedAt: new Date().toISOString(),
      };
    }
    await loadTaskList();
  } catch (e: any) {
    ElMessage.error(e.message || "取消任务失败");
  } finally {
    cancelLoading.value = false;
  }
}

/**
 * 下载任务结果
 * @param task 任务信息
 */
async function handleDownload(task: TaskVO) {
  downloadLoadingId.value = task.taskId;
  try {
    await taskStore.downloadResult(task.taskId);
    ElMessage.success("开始下载");
  } catch (e: any) {
    ElMessage.error(e.message || "下载失败");
  } finally {
    downloadLoadingId.value = null;
  }
}

/**
 * 页面可见性变化处理
 */
function handleVisibilityChange() {
  if (document.hidden) {
    // 页面不可见时暂停轮询
    taskStore.stopPolling();
  } else {
    // 页面恢复可见时，存在进行中任务则恢复轮询
    const hasActiveTasks = taskStore.taskList.some((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    if (hasActiveTasks) {
      taskStore.startPolling();
    }
  }
}

onMounted(() => {
  loadTaskList();
  document.addEventListener("visibilitychange", handleVisibilityChange);
});

onUnmounted(() => {
  taskStore.stopPolling();
  document.removeEventListener("visibilitychange", handleVisibilityChange);
});
</script>

<style lang="scss" scoped></style>
