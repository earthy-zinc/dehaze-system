/**
 * 算法选择扩展类型（RN 业务层补充）
 *
 * 基础算法类型复用 SDK 导出的 Algorithm，
 * 此处定义 Python 后端 /api/v1/algorithm-select/* 的请求/响应类型，
 * 与 dehaze-python app/models/schema/algorithm_select.py 保持一一对应。
 */
import type { Algorithm } from 'dehaze-sdk-js';

export type { Algorithm };

/** 智能推荐请求 */
export interface RecommendRequest {
  /** 待去雾图片 URL（后端可访问的远程地址） */
  imageUrl: string;
  /** 推荐数量，默认 3，范围 1-10 */
  topN?: number;
}

/** 智能推荐结果 VO */
export interface AlgorithmRecommendVO {
  algorithmId: number;
  algorithmName: string;
  /** 匹配得分（0-100） */
  score: number;
  /** 推荐理由 */
  reason?: string;
  /** 算法类型 */
  type?: string;
}

/** 收藏切换响应 */
export interface FavoriteToggleResult {
  favorited: boolean;
  favoriteId?: number;
}

/** 收藏 VO */
export interface FavoriteVO {
  id: number;
  userId: number;
  algorithmId: number;
  algorithmName?: string;
  createTime?: string;
}

/** 算法对比请求 */
export interface CompareRequest {
  /** 算法 ID 列表（2-4 个） */
  algorithmIds: number[];
  /** 待对比的图片 URL */
  imageUrl?: string;
}

/** 算法对比结果 VO */
export interface AlgorithmCompareVO {
  algorithmId: number;
  algorithmName: string;
  type?: string;
  /** 参数量 */
  params?: string;
  /** 计算量 */
  flops?: string;
  description?: string;
  status: number;
  /** 去雾结果 URL（如服务端已执行对比预测） */
  resultUrl?: string;
  /** 处理耗时（毫秒） */
  processTime?: number;
}
