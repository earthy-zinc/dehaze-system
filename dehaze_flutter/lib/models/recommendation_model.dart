import 'package:json_annotation/json_annotation.dart';

part 'recommendation_model.g.dart';

// ==================== 枚举 ====================

/// 推荐对象类型
enum RecommendationTargetType {
  @JsonValue('algorithm')
  algorithm,
  @JsonValue('preset')
  preset,
  @JsonValue('dataset')
  dataset,
  @JsonValue('enhance')
  enhance,
}

/// 雾霾浓度（30% 权重，影响算法强度选择）
enum HazeLevel {
  @JsonValue('light')
  light,
  @JsonValue('moderate')
  moderate,
  @JsonValue('heavy')
  heavy,
}

/// 场景类型（20% 权重）
enum SceneType {
  @JsonValue('urban')
  urban,
  @JsonValue('landscape')
  landscape,
  @JsonValue('building')
  building,
  @JsonValue('night')
  night,
  @JsonValue('backlight')
  backlight,
  @JsonValue('indoor')
  indoor,
}

/// 光照条件（15% 权重）
enum Lighting {
  @JsonValue('bright')
  bright,
  @JsonValue('normal')
  normal,
  @JsonValue('dark')
  dark,
  @JsonValue('veryDark')
  veryDark,
  @JsonValue('backlight')
  backlight,
}

/// 分辨率（5% 权重，影响算法可处理性）
enum Resolution {
  @JsonValue('sd')
  sd,
  @JsonValue('hd')
  hd,
  @JsonValue('uhd')
  uhd,
}

/// 噪声水平（10% 权重，影响预处理需求）
enum NoiseLevel {
  @JsonValue('low')
  low,
  @JsonValue('medium')
  medium,
  @JsonValue('high')
  high,
}

// ==================== 值对象 ====================

/// 颜色分布（10% 权重）
@JsonSerializable()
class ColorDistribution {
  const ColorDistribution({
    required this.temperature,
    required this.saturation,
  });

  factory ColorDistribution.fromJson(Map<String, dynamic> json) =>
      _$ColorDistributionFromJson(json);

  /// 色温
  final double temperature;

  /// 饱和度
  final double saturation;

  Map<String, dynamic> toJson() => _$ColorDistributionToJson(this);
}

// ==================== 请求对象 ====================

/// 图像分析请求（imageId 与 imageUrl 二选一）
@JsonSerializable()
class AnalyzeRequest {
  const AnalyzeRequest({this.imageId, this.imageUrl});

  factory AnalyzeRequest.fromJson(Map<String, dynamic> json) =>
      _$AnalyzeRequestFromJson(json);

  /// 已上传图片的 ID
  @JsonKey(name: 'imageId')
  final int? imageId;

  /// 图片 URL（与 imageId 二选一）
  @JsonKey(name: 'imageUrl')
  final String? imageUrl;

  Map<String, dynamic> toJson() => _$AnalyzeRequestToJson(this);
}

// ==================== 响应对象 ====================

/// 图像特征分析结果（7 维特征向量）
@JsonSerializable()
class ImageFeatureAnalysis {
  const ImageFeatureAnalysis({
    this.imageMd5,
    required this.hazeLevel,
    required this.hazeConfidence,
    required this.sceneType,
    required this.sceneConfidence,
    required this.lighting,
    required this.complexity,
    required this.colorDistribution,
    required this.resolution,
    required this.noiseLevel,
  });

  factory ImageFeatureAnalysis.fromJson(Map<String, dynamic> json) =>
      _$ImageFeatureAnalysisFromJson(json);

  /// 图像 MD5 值，用于关联推荐查询
  @JsonKey(name: 'imageMd5')
  final String? imageMd5;

  /// 雾霾浓度
  @JsonKey(name: 'hazeLevel')
  final HazeLevel hazeLevel;

  /// 雾霾浓度置信度 0-1
  @JsonKey(name: 'hazeConfidence')
  final double hazeConfidence;

  /// 场景类型
  @JsonKey(name: 'sceneType')
  final SceneType sceneType;

  /// 场景置信度 0-1
  @JsonKey(name: 'sceneConfidence')
  final double sceneConfidence;

  /// 光照条件
  final Lighting lighting;

  /// 图像复杂度 0-1（纹理丰富度、边缘密度）
  final double complexity;

  /// 颜色分布
  @JsonKey(name: 'colorDistribution')
  final ColorDistribution colorDistribution;

  /// 分辨率
  final Resolution resolution;

  /// 噪声水平
  @JsonKey(name: 'noiseLevel')
  final NoiseLevel noiseLevel;

  Map<String, dynamic> toJson() => _$ImageFeatureAnalysisToJson(this);
}

/// 推荐算法项
@JsonSerializable()
class RecommendedAlgorithm {
  const RecommendedAlgorithm({
    this.recommendationId,
    required this.algorithmId,
    required this.algorithmName,
    required this.matchScore,
    required this.reason,
    required this.rating,
    this.estimatedTime,
    this.effectDescription,
  });

  factory RecommendedAlgorithm.fromJson(Map<String, dynamic> json) =>
      _$RecommendedAlgorithmFromJson(json);

  /// 推荐记录 ID，用于提交反馈
  @JsonKey(name: 'recommendationId')
  final int? recommendationId;

  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  @JsonKey(name: 'algorithmName')
  final String algorithmName;

  /// 匹配度 0-100
  @JsonKey(name: 'matchScore')
  final int matchScore;

  /// 推荐理由
  final String reason;

  /// 算法评分 0-5（基于用户评价的综合评分）
  final int rating;

  /// 预估处理耗时(ms)
  @JsonKey(name: 'estimatedTime')
  final int? estimatedTime;

  /// 预期效果描述
  @JsonKey(name: 'effectDescription')
  final String? effectDescription;

  Map<String, dynamic> toJson() => _$RecommendedAlgorithmToJson(this);
}

/// 推荐结果（分析结果 + Top N 推荐算法列表）
@JsonSerializable()
class RecommendationResult {
  const RecommendationResult({
    required this.analysis,
    required this.recommendations,
  });

  factory RecommendationResult.fromJson(Map<String, dynamic> json) =>
      _$RecommendationResultFromJson(json);

  /// 图像特征分析结果
  final ImageFeatureAnalysis analysis;

  /// Top N 推荐算法列表（默认 Top 3）
  final List<RecommendedAlgorithm> recommendations;

  Map<String, dynamic> toJson() => _$RecommendationResultToJson(this);
}

// ==================== 反馈 ====================

/// 推荐反馈
@JsonSerializable()
class RecommendationFeedback {
  const RecommendationFeedback({
    required this.recommendationId,
    required this.useful,
  });

  factory RecommendationFeedback.fromJson(Map<String, dynamic> json) =>
      _$RecommendationFeedbackFromJson(json);

  @JsonKey(name: 'recommendationId')
  final int recommendationId;

  /// true=有用，false=无用
  final bool useful;

  Map<String, dynamic> toJson() => _$RecommendationFeedbackToJson(this);
}

// ==================== 规则 ====================

/// 推荐规则
@JsonSerializable()
class RecommendationRule {
  const RecommendationRule({
    this.id,
    required this.ruleName,
    required this.sceneType,
    required this.algorithmIds,
    required this.weight,
    required this.enabled,
  });

  factory RecommendationRule.fromJson(Map<String, dynamic> json) =>
      _$RecommendationRuleFromJson(json);

  final int? id;

  @JsonKey(name: 'ruleName')
  final String ruleName;

  @JsonKey(name: 'sceneType')
  final String sceneType;

  @JsonKey(name: 'algorithmIds')
  final List<int> algorithmIds;

  /// 权重 0-100
  final int weight;

  final bool enabled;

  Map<String, dynamic> toJson() => _$RecommendationRuleToJson(this);
}

// ==================== 报表 ====================

/// 推荐效果趋势条目
@JsonSerializable()
class RecommendationTrendItem {
  const RecommendationTrendItem({
    required this.date,
    required this.adoptionRate,
  });

  factory RecommendationTrendItem.fromJson(Map<String, dynamic> json) =>
      _$RecommendationTrendItemFromJson(json);

  final String date;

  @JsonKey(name: 'adoptionRate')
  final double adoptionRate;

  Map<String, dynamic> toJson() => _$RecommendationTrendItemToJson(this);
}

/// 推荐效果报表（管理员可见）
@JsonSerializable()
class RecommendationReport {
  const RecommendationReport({
    required this.totalRecommendations,
    required this.adoptionRate,
    required this.satisfactionRate,
    required this.coverageRate,
    required this.coldStartSuccessRate,
    required this.trend,
  });

  factory RecommendationReport.fromJson(Map<String, dynamic> json) =>
      _$RecommendationReportFromJson(json);

  /// 推荐总次数
  @JsonKey(name: 'totalRecommendations')
  final int totalRecommendations;

  /// 采纳率（目标 ≥ 40%）
  @JsonKey(name: 'adoptionRate')
  final double adoptionRate;

  /// 满意度（目标 ≥ 70%）
  @JsonKey(name: 'satisfactionRate')
  final double satisfactionRate;

  /// 覆盖率（目标 ≥ 60%）
  @JsonKey(name: 'coverageRate')
  final double coverageRate;

  /// 冷启动成功率（目标 ≥ 3 次）
  @JsonKey(name: 'coldStartSuccessRate')
  final double coldStartSuccessRate;

  /// 推荐效果趋势（按日/周/月聚合）
  final List<RecommendationTrendItem> trend;

  Map<String, dynamic> toJson() => _$RecommendationReportToJson(this);
}
