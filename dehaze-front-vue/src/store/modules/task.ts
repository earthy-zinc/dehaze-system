// 任务管理 store
import { DownloadTaskVO, ExportTaskAPI, TaskQuery } from "dehaze-sdk-js";

// 需要轮询的任务状态
const POLLING_STATUSES = ["pending", "processing"];
// 轮询间隔（毫秒）
const POLLING_INTERVAL = 3000;

export const useTaskStore = defineStore("task", () => {
  const taskList = ref<DownloadTaskVO[]>([]);
  const total = ref(0);
  const loading = ref(false);
  const currentTask = ref<DownloadTaskVO | null>(null);
  // 轮询定时器（内部私有，不导出）
  let pollingTimer: ReturnType<typeof setInterval> | null = null;
  // 轮询连续失败计数（避免接口持续失败时无限轮询）
  let pollingFailCount = 0;
  const MAX_POLLING_FAIL = 5;

  /** 获取任务列表 */
  const getTaskList = async (queryParams?: TaskQuery) => {
    loading.value = true;
    try {
      const data = await ExportTaskAPI.getList(queryParams);
      taskList.value = data.list;
      total.value = data.total;
    } finally {
      loading.value = false;
    }
  };

  /** 取消任务 */
  const cancelTask = async (taskId: string) => {
    await ExportTaskAPI.cancelTask(taskId);
  };

  /** 下载任务结果 */
  const downloadResult = async (taskId: string) => {
    const task = await ExportTaskAPI.getTaskStatus(taskId);
    if (task.status !== "completed") {
      throw new Error("任务未完成，无法下载");
    }
    if (!task.downloadUrl) {
      throw new Error("下载链接不存在");
    }
    if (task.expiresAt && new Date(task.expiresAt) < new Date()) {
      throw new Error("任务结果已过期");
    }
    const link = document.createElement("a");
    link.href = task.downloadUrl;
    link.download = "";
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  /** 轮询更新进行中任务的状态 */
  const pollTaskStatuses = async () => {
    const activeTasks = taskList.value.filter((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    if (activeTasks.length === 0) {
      stopPolling();
      return;
    }
    const results = await Promise.all(
      activeTasks.map((t) => ExportTaskAPI.getTaskStatus(t.taskId))
    );
    pollingFailCount = 0;
    results.forEach((updated, index) => {
      const taskId = activeTasks[index].taskId;
      const listIndex = taskList.value.findIndex((t) => t.taskId === taskId);
      if (listIndex !== -1) {
        taskList.value[listIndex] = updated;
      }
      if (currentTask.value?.taskId === taskId) {
        currentTask.value = updated;
      }
    });
  };

  /** 启动轮询 */
  const startPolling = () => {
    if (pollingTimer) return;
    pollingFailCount = 0;
    pollingTimer = setInterval(async () => {
      try {
        await pollTaskStatuses();
      } catch (e) {
        pollingFailCount += 1;
        if (pollingFailCount >= MAX_POLLING_FAIL) {
          console.error("任务状态轮询连续失败，已停止轮询", e);
          ElMessage.error("任务状态更新失败，已停止自动刷新");
          stopPolling();
        }
      }
    }, POLLING_INTERVAL);
  };

  /** 停止轮询 */
  const stopPolling = () => {
    if (pollingTimer) {
      clearInterval(pollingTimer);
      pollingTimer = null;
    }
  };

  return {
    taskList,
    total,
    loading,
    currentTask,
    getTaskList,
    cancelTask,
    downloadResult,
    startPolling,
    stopPolling,
  };
});
