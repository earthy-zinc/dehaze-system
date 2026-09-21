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
            {{ TASK_TYPE_LABELS[row.taskType] ?? row.taskType }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="statusTagType(row.status)">
              {{ statusLabel(row.status) }}
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
              v-if="row.status === 3 && row.taskCategory === 'export'"
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
            <el-tag :type="statusTagType(taskStore.currentTask.status)">
              {{ statusLabel(taskStore.currentTask.status) }}
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
import type { TagType } from "@/enums/TagType";
import { useTaskStore } from "@/store";
import {
  TASK_POLLING_STATUSES,
  TASK_STATUS_OPTIONS,
  TASK_TYPE_LABELS,
} from "../constants";
import { downloadByUrl } from "@/utils";

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

// 状态标签文本（单源派生）
function statusLabel(status: number): string | number {
  return TASK_STATUS_OPTIONS.find((o) => o.value === status)?.label ?? status;
}

// 状态标签类型（单源派生）
function statusTagType(status: number): TagType {
  return TASK_STATUS_OPTIONS.find((o) => o.value === status)?.tag ?? "info";
}

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
  return TASK_POLLING_STATUSES.includes(status);
}

/**
 * 加载任务列表
 */
async function loadTaskList() {
  try {
    await taskStore.getTaskList(queryParams);
  } catch {
    return;
  }
  // 存在进行中的任务时启动轮询，否则停止
  const hasActiveTasks = taskStore.taskList.some((t) =>
    TASK_POLLING_STATUSES.includes(t.status)
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
  } catch {
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
    const url = await taskStore.downloadResult(task.taskId);
    downloadByUrl(url);
    ElMessage.success("开始下载");
  } catch {
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
      TASK_POLLING_STATUSES.includes(t.status)
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
