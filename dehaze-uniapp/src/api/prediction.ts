/**
 * 预测/去雾处理 API
 *
 * API 路径与后端一致：
 * - POST /prediction       执行预测
 * - GET  /prediction/{id}  查询任务状态
 * - GET  /prediction/logs  预测日志（历史记录）
 */

import { get, post } from "./request";

// ==================== 类型定义 ====================

/** 预测请求 */
export interface PredictionForm {
  algorithmId: number;
  fileId?: number;
  imageUrl?: string;
  params?: string;
}

/** 预测结果 */
export interface PredictionResultVO {
  logId?: number;
  resultUrl: string;
  resultThumbnailUrl?: string;
  time: number;
  fromCache?: boolean;
}

/** 预测日志 */
export interface PredLogVO {
  id: number;
  algorithmId: number;
  algorithmName?: string;
  originUrl?: string;
  predUrl?: string;
  time?: number;
  createTime?: string;
}

/** 预测日志查询 */
export interface PredLogQuery {
  pageNum?: number;
  pageSize?: number;
  algorithmId?: number;
}

/** 分页结果 */
export interface PageResult<T> {
  list: T[];
  total: number;
  pageNum: number;
  pageSize: number;
}

// ==================== API 方法 ====================

/** 执行预测（去雾处理） */
export async function predict(
  data: PredictionForm
): Promise<PredictionResultVO> {
  return post<PredictionResultVO>(
    "/prediction",
    data as unknown as Record<string, unknown>
  );
}

/** 查询预测任务状态 */
export async function getPredictionStatus(
  taskId: number
): Promise<PredictionResultVO> {
  return get<PredictionResultVO>(`/prediction/${taskId}`);
}

/** 获取预测日志列表（历史记录） */
export async function getPredictionLogs(
  query?: PredLogQuery
): Promise<PageResult<PredLogVO>> {
  return get<PageResult<PredLogVO>>("/prediction/logs", {
    data: query as unknown as Record<string, unknown>,
  });
}
