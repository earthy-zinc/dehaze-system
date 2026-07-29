import type { ExportModule, ImportModule } from "dehaze-sdk-js";

export interface ImportExportToolbarProps {
  module: ExportModule;
  importable?: boolean;
  queryParams: object;
  extraImportParams?: Record<string, unknown>;
  onImportComplete?: () => void;
}

export interface ImportDialogProps {
  open: boolean;
  module: ImportModule;
  extraImportParams?: Record<string, unknown>;
  onClose: () => void;
  onImportComplete: () => void;
}

export interface ExportDialogProps {
  open: boolean;
  module: ExportModule;
  queryParams: object;
  initialFormat?: "excel" | "csv";
  onClose: () => void;
}

export interface TaskListDrawerProps {
  open: boolean;
  module?: ExportModule;
  onClose: () => void;
}

export const MODULE_LABEL_MAP: Record<ImportModule, string> = {
  user: "用户",
  role: "角色",
  dept: "部门",
  menu: "菜单",
  dict: "字典",
  algorithm: "算法",
};

export const TASK_TYPE_LABEL_MAP: Record<string, string> = {
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

export const STATUS_LABEL_MAP: Record<number, string> = {
  1: "待执行",
  2: "执行中",
  3: "已完成",
  4: "失败",
  5: "已取消",
};

export const STATUS_COLOR_MAP: Record<number, string> = {
  1: "blue",
  2: "blue",
  3: "green",
  4: "red",
  5: "default",
};
