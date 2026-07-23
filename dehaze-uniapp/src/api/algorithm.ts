/**
 * 算法选择扩展 API
 *
 * 算法 CRUD 直接使用 dehaze-sdk-js 的 AlgorithmAPI（Java 后端），
 * 本文件仅维护 SDK 未覆盖的算法选择扩展（推荐 / 收藏），
 * 通过 SDK 导出的 pythonService 直连 Python 后端。
 */

import { pythonService } from "dehaze-sdk-js";

// ==================== 算法选择扩展类型（Python 后端） ====================

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

// ==================== 算法选择扩展（Python 后端） ====================

/** 切换算法收藏状态（未收藏→添加，已收藏→取消） */
export function toggleAlgorithmFavorite(
  algorithmId: number
): Promise<ToggleFavoriteResult> {
  return pythonService.post("/api/v1/algorithm-select/favorite", {
    algorithmId,
  });
}

/** 获取当前用户的算法收藏列表 */
export function getAlgorithmFavorites(): Promise<AlgorithmFavorite[]> {
  return pythonService.get("/api/v1/algorithm-select/favorites");
}
