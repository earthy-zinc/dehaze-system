/**
 * 算法选择扩展 API
 *
 * SDK 的 AlgorithmAPI 未覆盖智能推荐/收藏/对比接口，
 * 此处通过 SDK 导出的 pythonService 直连 Python 后端 /api/v1/algorithm-select/*。
 */
import { pythonService } from 'dehaze-sdk-js';
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
    return pythonService.post('/api/v1/algorithm-select/recommend', data);
  }

  /** 切换收藏（未收藏→添加，已收藏→取消） */
  static toggleFavorite(algorithmId: number): Promise<FavoriteToggleResult> {
    return pythonService.post('/api/v1/algorithm-select/favorite', {
      algorithmId,
    });
  }

  /** 收藏列表 */
  static listFavorites(): Promise<FavoriteVO[]> {
    return pythonService.get('/api/v1/algorithm-select/favorites');
  }

  /** 算法对比（2-4 个） */
  static compare(
    algorithmIds: number[],
    imageUrl?: string,
  ): Promise<AlgorithmCompareVO[]> {
    return pythonService.post('/api/v1/algorithm-select/compare', {
      algorithmIds,
      imageUrl,
    });
  }
}

export default AlgorithmSelectAPI;
