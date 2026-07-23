import type { TaskStatus } from "dehaze-sdk-js";

// ==================== 常量定义 ====================

/** 轮询间隔（毫秒） */
export const POLLING_INTERVAL = 3000;

/** 需要轮询的任务状态 */
export const POLLING_STATUSES: TaskStatus[] = ["PENDING", "PROCESSING"];

/** 终态状态集合 */
export const TERMINAL_STATUSES: TaskStatus[] = [
  "COMPLETED",
  "FAILED",
  "CANCELLED",
];

/** 状态筛选选项 */
export const STATUS_FILTERS: { label: string; value: "" | TaskStatus }[] = [
  { label: "全部", value: "" },
  { label: "待执行", value: "PENDING" },
  { label: "执行中", value: "PROCESSING" },
  { label: "已完成", value: "COMPLETED" },
  { label: "失败", value: "FAILED" },
  { label: "已取消", value: "CANCELLED" },
];

/** 状态标签映射 */
export const STATUS_TAG: Record<
  TaskStatus,
  { label: string; color: "default" | "primary" | "success" | "danger" }
> = {
  PENDING: { label: "待执行", color: "primary" },
  PROCESSING: { label: "执行中", color: "primary" },
  COMPLETED: { label: "已完成", color: "success" },
  FAILED: { label: "失败", color: "danger" },
  CANCELLED: { label: "已取消", color: "default" },
};

/** 任务类型映射 */
export const TASK_TYPE_LABEL: Record<string, string> = {
  dataset_export: "数据集导出",
  item_download: "数据项下载",
  batch_download: "批量下载",
  custom_export: "自定义导出",
};

/** 每页条数 */
export const PAGE_SIZE = 10;

// ==================== 工具函数 ====================

/** 截断任务ID */
export function shortTaskId(taskId: string): string {
  if (taskId.length <= 16) return taskId;
  return `${taskId.slice(0, 8)}...${taskId.slice(-6)}`;
}
