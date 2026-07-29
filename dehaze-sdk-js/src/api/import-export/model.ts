/**
 * 导出请求参数（GET 查询参数或 POST 请求体）
 */
export interface ExportRequest {
  /** 文件格式：excel(默认) / csv */
  format?: "excel" | "csv";
  /** 是否强制异步：true / false / 不传(自动判断) */
  async?: boolean;
  /** 导出字段列表，不传则导出全部字段 */
  fields?: string[];
  /** 模块特定查询参数（如 keywords/status 等） */
  [key: string]: unknown;
}

/**
 * 异步导出任务创建结果
 */
export type ExportResult = {
  taskId: string;
  /** 任务状态（1=PENDING，与后端 TaskStatusEnum 对齐） */
  status: 1;
  estimatedCount: number;
};

/**
 * 导入请求参数（除 file 外的额外参数）
 */
export interface ImportRequest {
  /** 导入模式：all(全量,默认) / partial(部分) */
  mode?: "all" | "partial";
  /** 是否异步：true / false / 不传(数据量>1000行自动异步) */
  async?: boolean;
  /** 模块特定参数（如用户导入的 deptId） */
  [key: string]: unknown;
}

/**
 * 同步导入结果
 */
export interface ImportResult {
  totalRows: number;
  successCount: number;
  failureCount: number;
  skippedCount: number;
  errors: ImportError[];
  errorReportUrl?: string | null;
}

/**
 * 异步导入任务创建结果
 */
export interface ImportTaskResult {
  taskId: string;
  /** 任务状态（1=PENDING，与后端 TaskStatusEnum 对齐） */
  status: 1;
}

/**
 * 导入错误明细
 */
export interface ImportError {
  /** 行号（从 1 开始） */
  row: number;
  /** 出错字段名 */
  field?: string;
  /** 错误描述 */
  message: string;
}

/**
 * 支持导出的模块
 */
export type ExportModule = "user" | "role" | "dept" | "menu" | "dict" | "dataset" | "algorithm";

/**
 * 支持导入的模块（数据集不支持导入）
 */
export type ImportModule = "user" | "role" | "dept" | "menu" | "dict" | "algorithm";
