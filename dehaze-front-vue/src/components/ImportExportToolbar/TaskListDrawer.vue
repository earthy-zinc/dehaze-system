<script lang="ts" setup>
import { ExportModule, TaskVO } from "dehaze-sdk-js";
import { useTaskStore } from "@/store";

defineOptions({
  name: "TaskListDrawer",
});

const props = defineProps<{
  modelValue: boolean;
  module?: ExportModule;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
}>();

const visible = computed({
  get: () => props.modelValue,
  set: (val) => emit("update:modelValue", val),
});

const taskStore = useTaskStore();

const statusFilter = ref<number | "">("");
const categoryFilter = ref<"import" | "export" | "">("");

const POLLING_STATUSES = [1, 2];

const statusLabel: Record<number, string> = {
  1: "待执行",
  2: "执行中",
  3: "已完成",
  4: "失败",
  5: "已取消",
};

const statusTagType: Record<
  number,
  "info" | "primary" | "success" | "danger" | "warning"
> = {
  1: "info",
  2: "primary",
  3: "success",
  4: "danger",
  5: "warning",
};

const taskTypeLabel: Record<string, string> = {
  dataset_export: "数据集导出",
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

const queryParams = reactive({
  pageNum: 1,
  pageSize: 10,
});

const downloadLoadingId = ref<string | null>(null);
const cancelLoadingId = ref<string | null>(null);

function buildTaskTypeFilter(): string | undefined {
  if (props.module) {
    return [`${props.module}_export`, `${props.module}_import`].join(",");
  }
  return undefined;
}

async function loadTaskList() {
  const query: any = {
    ...queryParams,
    status: statusFilter.value || undefined,
    taskCategory: categoryFilter.value || undefined,
    taskType: buildTaskTypeFilter(),
  };
  try {
    await taskStore.getTaskList(query);
  } catch (e: any) {
    ElMessage.error(e.message || "加载任务列表失败");
    return;
  }
  const hasActiveTasks = taskStore.taskList.some((t) =>
    POLLING_STATUSES.includes(t.status)
  );
  if (hasActiveTasks) {
    taskStore.startPolling();
  } else {
    taskStore.stopPolling();
  }
}

function handleFilterChange() {
  queryParams.pageNum = 1;
  loadTaskList();
}

function canCancel(status: number): boolean {
  return POLLING_STATUSES.includes(status);
}

function progressStatus(
  status: number
): "" | "success" | "exception" | "warning" {
  if (status === 3) return "success";
  if (status === 4) return "exception";
  if (status === 5) return "warning";
  return "";
}

function formatTime(t?: string): string {
  if (!t) return "-";
  return new Date(t).toLocaleString("zh-CN");
}

async function handleCancel(task: TaskVO) {
  try {
    await ElMessageBox.confirm("确认取消该任务吗？", "提示", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    });
  } catch {
    return;
  }
  cancelLoadingId.value = task.taskId;
  try {
    await taskStore.cancelTask(task.taskId);
    ElMessage.success("任务已取消");
    await loadTaskList();
  } catch (e: any) {
    ElMessage.error(e.message || "取消任务失败");
  } finally {
    cancelLoadingId.value = null;
  }
}

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

function handleVisibilityChange() {
  if (document.hidden) {
    taskStore.stopPolling();
  } else {
    const hasActiveTasks = taskStore.taskList.some((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    if (hasActiveTasks) {
      taskStore.startPolling();
    }
  }
}

watch(visible, (val) => {
  if (val) {
    loadTaskList();
    document.addEventListener("visibilitychange", handleVisibilityChange);
  } else {
    taskStore.stopPolling();
    document.removeEventListener("visibilitychange", handleVisibilityChange);
  }
});

onUnmounted(() => {
  taskStore.stopPolling();
  document.removeEventListener("visibilitychange", handleVisibilityChange);
});
</script>

<template>
  <el-drawer
    v-model="visible"
    :title="
      module
        ? `${taskTypeLabel[module + '_export']?.replace('导出', '')}任务列表`
        : '任务列表'
    "
    size="900px"
    append-to-body
  >
    <div class="drawer-content">
      <div class="filter-bar">
        <el-radio-group v-model="categoryFilter" @change="handleFilterChange">
          <el-radio-button value="">全部</el-radio-button>
          <el-radio-button value="import">导入</el-radio-button>
          <el-radio-button value="export">导出</el-radio-button>
        </el-radio-group>
        <el-select
          v-model="statusFilter"
          placeholder="任务状态"
          clearable
          class="status-select"
          @change="handleFilterChange"
        >
          <el-option label="待执行" :value="1" />
          <el-option label="执行中" :value="2" />
          <el-option label="已完成" :value="3" />
          <el-option label="失败" :value="4" />
          <el-option label="已取消" :value="5" />
        </el-select>
        <el-button @click="loadTaskList">刷新</el-button>
      </div>

      <el-table
        v-loading="taskStore.loading"
        :data="taskStore.taskList"
        border
        empty-text="暂无任务"
      >
        <el-table-column
          label="任务ID"
          prop="taskId"
          width="260"
          show-overflow-tooltip
        />
        <el-table-column label="类型" width="120" align="center">
          <template #default="{ row }">
            {{ taskTypeLabel[row.taskType] ?? row.taskType }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="statusTagType[row.status] || 'info'">
              {{ statusLabel[row.status] ?? row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="进度" min-width="200">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="progressStatus(row.status)"
              :stroke-width="14"
              :text-inside="true"
            />
          </template>
        </el-table-column>
        <el-table-column label="创建时间" width="170" align="center">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="170" align="center" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="canCancel(row.status)"
              type="warning"
              link
              :loading="cancelLoadingId === row.taskId"
              @click="handleCancel(row as TaskVO)"
            >
              取消
            </el-button>
            <el-button
              v-if="row.status === 3"
              type="success"
              link
              :loading="downloadLoadingId === row.taskId"
              @click="handleDownload(row as TaskVO)"
            >
              下载
            </el-button>
            <span v-if="!canCancel(row.status) && row.status !== 3">-</span>
          </template>
        </el-table-column>
      </el-table>

      <Pagination
        v-model:page="queryParams.pageNum"
        v-model:limit="queryParams.pageSize"
        :total="taskStore.total"
        @pagination="loadTaskList"
      />
    </div>
  </el-drawer>
</template>

<style lang="scss" scoped>
.drawer-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
}

.filter-bar {
  display: flex;
  gap: 12px;
  align-items: center;
}

.status-select {
  width: 140px;
}
</style>
