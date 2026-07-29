/**
 * 任务状态信息（标签 + 颜色）
 */
import type { TaskCategory, TaskVO, TaskStatus } from 'dehaze-sdk-js';
import { theme } from '@/theme';

/** 任务（SDK TaskVO 别名） */
export type Task = TaskVO;

export type { TaskStatus, TaskCategory };

export const TaskStatusEnum = {
  PENDING: 1,
  PROCESSING: 2,
  COMPLETED: 3,
  FAILED: 4,
  CANCELLED: 5,
} as const satisfies Record<string, TaskStatus>;

export const TASK_STATUS_MAP: Record<
  TaskStatus,
  { label: string; color: string; bgColor: string }
> = {
  1: { label: '待执行', color: theme.colors.status.info, bgColor: `${theme.colors.status.info}15` },
  2: { label: '执行中', color: theme.colors.status.info, bgColor: `${theme.colors.status.info}15` },
  3: { label: '已完成', color: theme.colors.status.success, bgColor: `${theme.colors.status.success}15` },
  4: { label: '失败', color: theme.colors.status.error, bgColor: `${theme.colors.status.error}15` },
  5: { label: '已取消', color: theme.colors.text.secondary, bgColor: `${theme.colors.text.secondary}15` },
};

/** 任务类别筛选选项 */
export const CATEGORY_FILTERS: { label: string; value: 'ALL' | TaskCategory }[] = [
  { label: '全部', value: 'ALL' },
  { label: '导入', value: 'import' },
  { label: '导出', value: 'export' },
];

export const TASK_TYPE_MAP: Record<string, string> = {
  dataset_export: '数据集导出',
  user_export: '用户导出',
  role_export: '角色导出',
  dept_export: '部门导出',
  menu_export: '菜单导出',
  dict_export: '字典导出',
  algorithm_export: '算法导出',
  user_import: '用户导入',
  role_import: '角色导入',
  dept_import: '部门导入',
  menu_import: '菜单导入',
  dict_import: '字典导入',
  algorithm_import: '算法导入',
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
