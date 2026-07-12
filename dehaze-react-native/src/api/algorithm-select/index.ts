/**
 * 算法选择扩展 API
 *
 * SDK 的 AlgorithmAPI 未覆盖智能推荐/收藏/对比接口，
 * 此处通过 SDK 导出的 pythonService 直连 Python 后端 /api/v1/algorithm-select/*。
 */
import { pythonService } from 'dehaze-sdk-js';
import type {
  RecommendRequest,
  RecommendResult,
  CompareResult,
} from '@/types/algorithm';

class AlgorithmSelectAPI {
  /** 智能推荐 Top3 */
  static recommend(data: RecommendRequest): Promise<RecommendResult[]> {
    return pythonService.post('/api/v1/algorithm-select/recommend', data);
  }

  /** 切换收藏 */
  static toggleFavorite(algorithmId: number): Promise<{ favorite: boolean }> {
    return pythonService.post('/api/v1/algorithm-select/favorite', {
      algorithmId,
    });
  }

  /** 收藏列表 */
  static listFavorites(): Promise<RecommendResult[]> {
    return pythonService.get('/api/v1/algorithm-select/favorites');
  }

  /** 算法对比（最多 3 个） */
  static compare(algorithmIds: number[]): Promise<CompareResult[]> {
    return pythonService.post('/api/v1/algorithm-select/compare', {
      algorithmIds,
    });
  }
}

export default AlgorithmSelectAPI;
