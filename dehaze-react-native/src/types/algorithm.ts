/**
 * 算法相关类型（RN 业务层补充）
 *
 * 基础算法类型复用 SDK 导出的 Algorithm，
 * 此处仅定义 SDK 未覆盖的推荐/对比业务类型。
 */
import type { Algorithm } from 'dehaze-sdk-js';

export type { Algorithm };

export interface RecommendRequest {
  imageUrl?: string;
  imageBase64?: string;
  features?: {
    hazeDensity?: number;
    sceneType?: string;
    lighting?: string;
  };
}

export interface RecommendResult {
  algorithm: Algorithm;
  score: number;
  reason?: string;
}

export interface CompareResult {
  algorithm: Algorithm;
  metrics?: {
    psnr?: number;
    ssim?: number;
    speed?: number;
    rating?: number;
  };
}
