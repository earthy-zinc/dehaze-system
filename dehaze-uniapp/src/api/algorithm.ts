/**
 * 算法管理 API
 *
 * API 路径与后端一致：
 * - GET    /algorithms          算法列表
 * - GET    /algorithms/options  算法下拉选项
 * - GET    /algorithms/{id}     算法详情
 */

import { get } from "./request";

// ==================== 类型定义 ====================

/** 算法模型 */
export interface Algorithm {
  id: number;
  parentId: number;
  name: string;
  type: string;
  description: string;
  img?: string;
  path?: string;
  importPath?: string;
  params?: string;
  flops?: string;
  status?: number;
  size?: string;
  version?: string;
  createTime?: string;
  children?: Algorithm[];
}

/** 算法查询参数 */
export interface AlgorithmQuery {
  keywords?: string;
}

/** 下拉选项 */
export interface AlgorithmOption {
  value: number;
  label: string;
}

// ==================== API 方法 ====================

/** 获取算法列表 */
export async function getAlgorithmList(query?: AlgorithmQuery): Promise<Algorithm[]> {
  return get<Algorithm[]>("/algorithms", {
    data: query as Record<string, unknown>,
  });
}

/** 获取算法下拉选项 */
export async function getAlgorithmOptions(): Promise<AlgorithmOption[]> {
  return get<AlgorithmOption[]>("/algorithms/options");
}

/** 获取算法详情 */
export async function getAlgorithmDetail(id: number): Promise<Algorithm> {
  return get<Algorithm>(`/algorithms/${id}`);
}
