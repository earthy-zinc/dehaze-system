/**
 * 任务状态信息（标签 + 颜色）
 */
import type { TaskVO, TaskStatus } from 'dehaze-sdk-js';

/** 任务（SDK TaskVO 别名） */
export type Task = TaskVO;

export type { TaskStatus };

/** 任务状态枚举值（统一常量，避免散落的裸字符串） */
export const TaskStatusEnum = {
  PENDING: 'PENDING',
  PROCESSING: 'PROCESSING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
  CANCELLED: 'CANCELLED',
} as const satisfies Record<TaskStatus, TaskStatus>;

export const TASK_STATUS_MAP: Record<
  TaskStatus,
  { label: string; color: string; bgColor: string }
> = {
  PENDING: { label: '待执行', color: '#1890ff', bgColor: '#e6f7ff' },
  PROCESSING: { label: '执行中', color: '#1890ff', bgColor: '#e6f7ff' },
  COMPLETED: { label: '已完成', color: '#52c41a', bgColor: '#f6ffed' },
  FAILED: { label: '失败', color: '#ff4d4f', bgColor: '#fff2f0' },
  CANCELLED: { label: '已取消', color: '#8c8c8c', bgColor: '#fafafa' },
};

export const TASK_TYPE_MAP: Record<string, string> = {
  dataset_export: '数据集导出',
  item_download: '数据项下载',
  batch_download: '批量下载',
  custom_export: '自定义导出',
};

/** 终态：不再轮询 */
export const TERMINAL_STATUSES: TaskStatus[] = [
  TaskStatusEnum.COMPLETED,
  TaskStatusEnum.FAILED,
  TaskStatusEnum.CANCELLED,
];

/** 可取消的状态（未进入终态前） */
export const CANCELLABLE_STATUSES: TaskStatus[] = [
  TaskStatusEnum.PENDING,
  TaskStatusEnum.PROCESSING,
];

export function isTerminal(task?: Task | null): boolean {
  return !!task && TERMINAL_STATUSES.includes(task.status);
}

export function isCancellable(task?: Task | null): boolean {
  return !!task && CANCELLABLE_STATUSES.includes(task.status);
}

export function formatTaskTime(time?: string): string {
  if (!time) return '-';
  try {
    const d = new Date(time);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  } catch {
    return time;
  }
}
