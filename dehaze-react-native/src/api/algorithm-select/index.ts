import { service } from 'dehaze-sdk-js';
import type {
  AlgorithmCompareVO,
  AlgorithmRecommendVO,
  FavoriteToggleResult,
  FavoriteVO,
  RecommendRequest,
} from '@/types/algorithm';

class AlgorithmSelectAPI {
  /** 智能推荐 Top N（默认 3） */
  static recommend(data: RecommendRequest): Promise<AlgorithmRecommendVO[]> {
    return service.post('/api/v1/algorithm-select/recommend', data);
  }

  /** 切换收藏（未收藏→添加，已收藏→取消） */
  static toggleFavorite(algorithmId: number): Promise<FavoriteToggleResult> {
    return service.post('/api/v1/algorithm-select/favorite', {
      algorithmId,
    });
  }

  /** 收藏列表 */
  static listFavorites(): Promise<FavoriteVO[]> {
    return service.get('/api/v1/algorithm-select/favorites');
  }

  /** 算法对比（2-4 个） */
  static compare(
    algorithmIds: number[],
    imageUrl?: string,
  ): Promise<AlgorithmCompareVO[]> {
    return service.post('/api/v1/algorithm-select/compare', {
      algorithmIds,
      imageUrl,
    });
  }
}

export default AlgorithmSelectAPI;
