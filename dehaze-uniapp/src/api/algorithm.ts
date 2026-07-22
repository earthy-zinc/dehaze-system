/**
 * 算法管理 API
 *
 * API 路径与后端一致：
 * - GET    /algorithms                       算法列表
 * - GET    /algorithms/options               算法下拉选项
 * - GET    /algorithms/{id}                  算法详情
 * - POST   /algorithm-select/favorite        收藏/取消收藏算法
 * - GET    /algorithm-select/favorites       收藏列表
 * - POST   /algorithm-select/recommend       智能推荐算法
 * - POST   /algorithm-select/compare         算法对比
 */

import { get, post } from "./request";

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

/** 收藏记录 */
export interface AlgorithmFavorite {
  id: number;
  userId: number;
  algorithmId: number;
  algorithmName?: string;
  createTime?: string;
}

/** 切换收藏结果 */
export interface ToggleFavoriteResult {
  favorited: boolean;
  favoriteId?: number;
}

/** 智能推荐请求 */
export interface RecommendRequest {
  imageUrl: string;
  topN?: number;
}

/** 算法推荐结果 */
export interface AlgorithmRecommendVO {
  algorithmId: number;
  algorithmName: string;
  score: number;
  reason: string;
  type?: string;
}

// ==================== API 方法 ====================

/** 获取算法列表 */
export async function getAlgorithmList(
  query?: AlgorithmQuery
): Promise<Algorithm[]> {
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

/** 切换算法收藏状态（未收藏→添加，已收藏→取消） */
export async function toggleAlgorithmFavorite(
  algorithmId: number
): Promise<ToggleFavoriteResult> {
  return post<ToggleFavoriteResult>("/algorithm-select/favorite", {
    algorithmId,
  });
}

/** 获取当前用户的算法收藏列表 */
export async function getAlgorithmFavorites(): Promise<AlgorithmFavorite[]> {
  return get<AlgorithmFavorite[]>("/algorithm-select/favorites");
}

/** 智能推荐算法 */
export async function recommendAlgorithms(
  data: RecommendRequest
): Promise<AlgorithmRecommendVO[]> {
  return post<AlgorithmRecommendVO[]>(
    "/algorithm-select/recommend",
    data as unknown as Record<string, unknown>
  );
}
