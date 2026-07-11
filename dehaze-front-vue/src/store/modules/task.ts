// 任务管理小仓库
import { DownloadTaskVO, ExportTaskAPI, TaskQuery } from "dehaze-sdk-js";

// 需要轮询的任务状态
const POLLING_STATUSES = ["pending", "processing"];
// 轮询间隔（毫秒）
const POLLING_INTERVAL = 3000;

export const useTaskStore = defineStore("task", () => {
  // 任务列表
  const taskList = ref<DownloadTaskVO[]>([]);
  // 任务总数
  const total = ref(0);
  // 列表加载状态
  const loading = ref(false);
  // 当前查看的任务
  const currentTask = ref<DownloadTaskVO | null>(null);
  // 轮询定时器
  const pollingTimer = ref<ReturnType<typeof setInterval> | null>(null);

  /**
   * 获取任务列表
   * @param queryParams 查询参数
   */
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

  /**
   * 查询任务状态
   * @param taskId 任务ID
   * @returns 任务信息
   */
  const getTaskStatus = async (taskId: string) => {
    return await ExportTaskAPI.getTaskStatus(taskId);
  };

  /**
   * 取消任务
   * @param taskId 任务ID
   */
  const cancelTask = async (taskId: string) => {
    await ExportTaskAPI.cancelTask(taskId);
  };

  /**
   * 下载任务结果
   * @param taskId 任务ID
   */
  const downloadResult = async (taskId: string) => {
    const task = await ExportTaskAPI.getTaskStatus(taskId);
    if (task.status !== "completed") {
      throw new Error("任务未完成，无法下载");
    }
    if (!task.downloadUrl) {
      throw new Error("下载链接不存在");
    }
    // 检查结果是否过期
    if (task.expiresAt && new Date(task.expiresAt) < new Date()) {
      throw new Error("任务结果已过期");
    }
    // 触发浏览器下载
    const link = document.createElement("a");
    link.href = task.downloadUrl;
    link.download = "";
    link.target = "_blank";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  /**
   * 轮询更新进行中任务的状态
   */
  const pollTaskStatuses = async () => {
    const activeTasks = taskList.value.filter((t) =>
      POLLING_STATUSES.includes(t.status)
    );
    // 没有需要轮询的任务，停止轮询
    if (activeTasks.length === 0) {
      stopPolling();
      return;
    }
    // 并行查询所有进行中任务的状态
    const results = await Promise.all(
      activeTasks.map((t) =>
        ExportTaskAPI.getTaskStatus(t.taskId).catch(() => null)
      )
    );
    // 更新列表中对应任务的状态
    results.forEach((updated, index) => {
      if (!updated) return;
      const taskId = activeTasks[index].taskId;
      const listIndex = taskList.value.findIndex((t) => t.taskId === taskId);
      if (listIndex !== -1) {
        taskList.value[listIndex] = updated;
      }
      // 同步更新当前查看的任务
      if (currentTask.value?.taskId === taskId) {
        currentTask.value = updated;
      }
    });
  };

  /**
   * 启动轮询
   */
  const startPolling = () => {
    if (pollingTimer.value) return;
    pollingTimer.value = setInterval(pollTaskStatuses, POLLING_INTERVAL);
  };

  /**
   * 停止轮询
   */
  const stopPolling = () => {
    if (pollingTimer.value) {
      clearInterval(pollingTimer.value);
      pollingTimer.value = null;
    }
  };

  return {
    taskList,
    total,
    loading,
    currentTask,
    pollingTimer,
    getTaskList,
    getTaskStatus,
    cancelTask,
    downloadResult,
    startPolling,
    stopPolling,
  };
});
