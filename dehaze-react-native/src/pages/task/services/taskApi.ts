/**
 * 任务管理 API 封装
 *
 * 统一代理到 SDK TaskAPI，任务来源包括：
 * - 数据集导出（DatasetAPI.createExportTask）
 * - 数据项下载（DatasetItemAPI.createDownloadTask / batchDownload）
 * - 通用任务创建（TaskAPI.create）
 */
import { TaskAPI } from 'dehaze-sdk-js';
import type { TaskCreateForm, TaskQuery, PageResult } from 'dehaze-sdk-js';
import type { Task } from '../types';

export type TaskPage = PageResult<Task[]>;

export const taskApi = {
  /** 创建任务 */
  create(data: TaskCreateForm): Promise<Task> {
    return TaskAPI.create(data);
  },

  /** 查询任务状态 */
  getStatus(taskId: string): Promise<Task> {
    return TaskAPI.getStatus(taskId);
  },

  /** 任务列表分页查询 */
  getPage(query?: TaskQuery): Promise<TaskPage> {
    return TaskAPI.getPage(query);
  },

  /** 取消任务 */
  cancel(taskId: string): Promise<void> {
    return TaskAPI.cancel(taskId);
  },

  /**
   * 下载任务结果（返回 Blob）
   * RN 端使用 fetch + 文件系统保存，调用方需自行处理 Blob
   */
  download(taskId: string): Promise<Blob> {
    return TaskAPI.download(taskId);
  },
};
