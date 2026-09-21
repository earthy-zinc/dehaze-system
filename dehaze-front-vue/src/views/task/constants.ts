import type { TagType } from "@/enums/TagType";

/**
 * 任务状态单源定义（三端 TaskStatus 对齐：1待执行/2执行中/3已完成/4失败/5已取消）
 */
export const TASK_STATUS_OPTIONS: {
  value: number;
  label: string;
  tag: TagType;
}[] = [
  { value: 1, label: "待执行", tag: "info" },
  { value: 2, label: "执行中", tag: "primary" },
  { value: 3, label: "已完成", tag: "success" },
  { value: 4, label: "失败", tag: "danger" },
  { value: 5, label: "已取消", tag: "warning" },
];

/**
 * 任务类型 → 展示名（与三端 TaskType 枚举对齐）
 */
export const TASK_TYPE_LABELS: Record<string, string> = {
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

/**
 * 需要轮询进度的任务状态（待执行/执行中）
 */
export const TASK_POLLING_STATUSES = [1, 2];
