import { AnalyzeRequest, RecommendationFeedback, RecommendationRule } from "../../index";
import { DEHAZE_HOST } from "#/config/constant";
import { uniqueName } from "./common";

const DATASET_BASE_URL = `http://${DEHAZE_HOST}:9000/datasets`;

/**
 * 创建图像分析请求
 * 默认使用 NH-HAZE-2023 城市雾霾图片 URL（与 model factory 保持一致）
 */
export function createAnalyzeRequest(overrides: Partial<AnalyzeRequest> = {}): AnalyzeRequest {
  return {
    imageUrl: `${DATASET_BASE_URL}/NH-HAZE-2023/hazy/001.JPG`,
    ...overrides,
  };
}

/** 创建推荐反馈（默认有用反馈） */
export function createFeedback(
  overrides: Partial<RecommendationFeedback> = {}
): RecommendationFeedback {
  return {
    recommendationId: 1,
    useful: true,
    ...overrides,
  };
}

/** 创建推荐规则配置 */
export function createRule(overrides: Partial<RecommendationRule> = {}): RecommendationRule {
  return {
    ruleName: uniqueName("测试规则"),
    sceneType: "urban",
    algorithmIds: [1, 2, 3],
    weight: 50,
    enabled: true,
    ...overrides,
  };
}
