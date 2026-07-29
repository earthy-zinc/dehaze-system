import type { TaskCategory, TaskStatus } from "dehaze-sdk-js";

// ==================== 常量定义 ====================

/** 轮询间隔（毫秒） */
export const POLLING_INTERVAL = 3000;

/** 需要轮询的任务状态 */
export const POLLING_STATUSES: TaskStatus[] = [1, 2];

/** 终态状态集合 */
export const TERMINAL_STATUSES: TaskStatus[] = [3, 4, 5];

/** 状态筛选选项 */
export const STATUS_FILTERS: { label: string; value: "" | TaskStatus }[] = [
  { label: "全部", value: "" },
  { label: "待执行", value: 1 },
  { label: "执行中", value: 2 },
  { label: "已完成", value: 3 },
  { label: "失败", value: 4 },
  { label: "已取消", value: 5 },
];

/** 任务类别筛选选项 */
export const CATEGORY_FILTERS: { label: string; value: "" | TaskCategory }[] = [
  { label: "全部", value: "" },
  { label: "导入", value: "import" },
  { label: "导出", value: "export" },
];

/** 状态标签映射 */
export const STATUS_TAG: Record<
  TaskStatus,
  { label: string; color: "default" | "primary" | "success" | "danger" }
> = {
  1: { label: "待执行", color: "primary" },
  2: { label: "执行中", color: "primary" },
  3: { label: "已完成", color: "success" },
  4: { label: "失败", color: "danger" },
  5: { label: "已取消", color: "default" },
};

/** 任务类型映射 */
export const TASK_TYPE_LABEL: Record<string, string> = {
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

/** 每页条数 */
export const PAGE_SIZE = 10;

// ==================== 工具函数 ====================

/** 截断任务ID */
export function shortTaskId(taskId: string): string {
  if (taskId.length <= 16) return taskId;
  return `${taskId.slice(0, 8)}...${taskId.slice(-6)}`;
}
