/**
 * 推荐管理模块类型定义
 * 对应 API 接口文档：03-模块设计/基础模块/推荐管理/API接口.md
 */

/** 推荐对象类型 */
export type RecommendationTargetType = "algorithm" | "preset" | "dataset" | "enhance";

/** 雾霾浓度（30% 权重，影响算法强度选择） */
export type HazeLevel = "light" | "moderate" | "heavy";

/** 场景类型（20% 权重） */
export type SceneType = "urban" | "landscape" | "building" | "night" | "backlight" | "indoor";

/** 光照条件（15% 权重） */
export type Lighting = "bright" | "normal" | "dark" | "veryDark" | "backlight";

/** 分辨率（5% 权重，影响算法可处理性） */
export type Resolution = "sd" | "hd" | "uhd";

/** 噪声水平（10% 权重，影响预处理需求） */
export type NoiseLevel = "low" | "medium" | "high";

/** 颜色分布（10% 权重） */
export interface ColorDistribution {
  /** 色温 */
  temperature: number;
  /** 饱和度 */
  saturation: number;
}

/**
 * 图像特征分析结果（7 维特征向量）
 * 作为推荐匹配的输入，对应需求规格 F-REC-001
 */
export interface ImageFeatureAnalysis {
  /** 图像MD5值（基于imageUrl计算），用于关联推荐查询 */
  imageMd5?: string;
  /** 雾霾浓度 */
  hazeLevel: HazeLevel;
  /** 雾霾浓度置信度 0-1 */
  hazeConfidence: number;
  /** 场景类型 */
  sceneType: SceneType;
  /** 场景置信度 0-1 */
  sceneConfidence: number;
  /** 光照条件 */
  lighting: Lighting;
  /** 图像复杂度 0-1（纹理丰富度、边缘密度） */
  complexity: number;
  /** 颜色分布 */
  colorDistribution: ColorDistribution;
  /** 分辨率 */
  resolution: Resolution;
  /** 噪声水平 */
  noiseLevel: NoiseLevel;
}

/** 图像分析请求（imageId 与 imageUrl 二选一） */
export interface AnalyzeRequest {
  /** 已上传图片的 ID */
  imageId?: number;
  /** 图片 URL（与 imageId 二选一） */
  imageUrl?: string;
}

/** 推荐算法项 */
export interface RecommendedAlgorithm {
  /** 推荐记录ID（sys_recommendation.id），用于提交反馈 */
  recommendationId?: number;
  algorithmId: number;
  algorithmName: string;
  /** 匹配度 0-100 */
  matchScore: number;
  /** 推荐理由（一句话说明为什么推荐） */
  reason: string;
  /** 算法评分 0-5（基于用户评价的综合评分） */
  rating: number;
  /** 预估处理耗时(ms) */
  estimatedTime?: number;
  /** 预期效果描述 */
  effectDescription?: string;
}

/** 推荐结果（分析结果 + Top N 推荐算法列表） */
export interface RecommendationResult {
  /** 图像特征分析结果 */
  analysis: ImageFeatureAnalysis;
  /** Top N 推荐算法列表（默认 Top 3） */
  recommendations: RecommendedAlgorithm[];
}

/** 推荐反馈 */
export interface RecommendationFeedback {
  recommendationId: number;
  /** true=有用，false=无用 */
  useful: boolean;
}

/** 推荐规则 */
export interface RecommendationRule {
  id?: number;
  ruleName: string;
  sceneType: string;
  algorithmIds: number[];
  /** 权重 0-100 */
  weight: number;
  enabled: boolean;
}

/** 推荐效果趋势条目 */
export interface RecommendationTrendItem {
  date: string;
  adoptionRate: number;
}

/** 推荐效果报表（管理员可见，对应需求规格 5.1 核心指标） */
export interface RecommendationReport {
  /** 推荐总次数 */
  totalRecommendations: number;
  /** 采纳率（目标 ≥ 40%） */
  adoptionRate: number;
  /** 满意度（目标 ≥ 70%） */
  satisfactionRate: number;
  /** 覆盖率（目标 ≥ 60%） */
  coverageRate: number;
  /** 冷启动成功率（目标 ≥ 3 次） */
  coldStartSuccessRate: number;
  /** 推荐效果趋势（按日/周/月聚合） */
  trend: RecommendationTrendItem[];
}
