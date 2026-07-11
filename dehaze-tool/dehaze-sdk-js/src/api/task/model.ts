import { PageQuery } from "@/types";

/**
 * 任务类型
 */
export type TaskType =
  | "dataset_export"
  | "item_download"
  | "batch_download"
  | "custom_export";

/**
 * 任务状态
 */
export type TaskStatus =
  | "PENDING"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

/**
 * 任务创建表单
 */
export interface TaskCreateForm {
  /** 任务类型 */
  type: TaskType;
  /** 目标资源ID（导出单个资源时使用） */
  targetId?: number;
  /** 目标资源ID列表（批量导出时使用） */
  targetIds?: number[];
  /** 导出选项配置（文件组织方式、包含类型等） */
  options?: Record<string, any>;
}

/**
 * 任务查询参数
 */
export interface TaskQuery extends PageQuery {
  /** 任务状态筛选 */
  status?: TaskStatus;
  /** 任务类型筛选 */
  taskType?: TaskType;
}

/**
 * 任务信息
 */
export interface TaskVO {
  /** 任务ID */
  taskId: string;
  /** 任务状态 */
  status: TaskStatus;
  /** 进度百分比（0-100） */
  progress: number;
  /** 任务类型 */
  taskType?: TaskType;
  /** 总文件数 */
  totalFiles?: number;
  /** 已处理文件数 */
  processedFiles?: number;
  /** 下载链接（任务完成时返回） */
  downloadUrl?: string;
  /** 过期时间 */
  expiresAt?: string;
  /** 创建时间 */
  createdAt?: string;
  /** 开始执行时间 */
  startedAt?: string;
  /** 完成时间 */
  completedAt?: string;
  /** 错误信息（失败时返回） */
  error?: string;
}
