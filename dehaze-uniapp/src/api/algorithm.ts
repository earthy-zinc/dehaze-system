/**
 * 算法选择扩展 API
 *
 * SDK 的 AlgorithmAPI 未覆盖智能推荐/收藏接口，
 * 此处通过 SDK 导出的 service 直连后端。
 */

import { service } from "dehaze-sdk-js";

export interface AlgorithmFavorite {
  id: number;
  userId: number;
  algorithmId: number;
  algorithmName?: string;
  createTime?: string;
}

export interface ToggleFavoriteResult {
  favorited: boolean;
  favoriteId?: number;
}

export interface RecommendRequest {
  imageUrl: string;
  topN?: number;
}

export interface AlgorithmRecommendVO {
  algorithmId: number;
  algorithmName: string;
  score: number;
  reason: string;
  type?: string;
}

export function toggleAlgorithmFavorite(
  algorithmId: number
): Promise<ToggleFavoriteResult> {
  return service.post("/api/v1/algorithm-select/favorite", {
    algorithmId,
  });
}

export function getAlgorithmFavorites(): Promise<AlgorithmFavorite[]> {
  return service.get("/api/v1/algorithm-select/favorites");
}

export function recommendAlgorithms(
  imageUrl: string,
  topN?: number
): Promise<AlgorithmRecommendVO[]> {
  return service.post("/api/v1/algorithm-select/recommend", {
    imageUrl,
    topN,
  });
}
