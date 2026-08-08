// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'recommendation_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ColorDistribution _$ColorDistributionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ColorDistribution', json, ($checkedConvert) {
      final val = ColorDistribution(
        temperature: $checkedConvert(
          'temperature',
          (v) => (v as num).toDouble(),
        ),
        saturation: $checkedConvert('saturation', (v) => (v as num).toDouble()),
      );
      return val;
    });

Map<String, dynamic> _$ColorDistributionToJson(ColorDistribution instance) =>
    <String, dynamic>{
      'temperature': instance.temperature,
      'saturation': instance.saturation,
    };

AnalyzeRequest _$AnalyzeRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnalyzeRequest', json, ($checkedConvert) {
      final val = AnalyzeRequest(
        imageId: $checkedConvert('imageId', (v) => (v as num?)?.toInt()),
        imageUrl: $checkedConvert('imageUrl', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AnalyzeRequestToJson(AnalyzeRequest instance) =>
    <String, dynamic>{
      if (instance.imageId case final value?) 'imageId': value,
      if (instance.imageUrl case final value?) 'imageUrl': value,
    };

ImageFeatureAnalysis _$ImageFeatureAnalysisFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageFeatureAnalysis', json, ($checkedConvert) {
  final val = ImageFeatureAnalysis(
    imageMd5: $checkedConvert('imageMd5', (v) => v as String?),
    hazeLevel: $checkedConvert(
      'hazeLevel',
      (v) => $enumDecode(_$HazeLevelEnumMap, v),
    ),
    hazeConfidence: $checkedConvert(
      'hazeConfidence',
      (v) => (v as num).toDouble(),
    ),
    sceneType: $checkedConvert(
      'sceneType',
      (v) => $enumDecode(_$SceneTypeEnumMap, v),
    ),
    sceneConfidence: $checkedConvert(
      'sceneConfidence',
      (v) => (v as num).toDouble(),
    ),
    lighting: $checkedConvert(
      'lighting',
      (v) => $enumDecode(_$LightingEnumMap, v),
    ),
    complexity: $checkedConvert('complexity', (v) => (v as num).toDouble()),
    colorDistribution: $checkedConvert(
      'colorDistribution',
      (v) => ColorDistribution.fromJson(v as Map<String, dynamic>),
    ),
    resolution: $checkedConvert(
      'resolution',
      (v) => $enumDecode(_$ResolutionEnumMap, v),
    ),
    noiseLevel: $checkedConvert(
      'noiseLevel',
      (v) => $enumDecode(_$NoiseLevelEnumMap, v),
    ),
  );
  return val;
});

Map<String, dynamic> _$ImageFeatureAnalysisToJson(
  ImageFeatureAnalysis instance,
) => <String, dynamic>{
  if (instance.imageMd5 case final value?) 'imageMd5': value,
  'hazeLevel': _$HazeLevelEnumMap[instance.hazeLevel]!,
  'hazeConfidence': instance.hazeConfidence,
  'sceneType': _$SceneTypeEnumMap[instance.sceneType]!,
  'sceneConfidence': instance.sceneConfidence,
  'lighting': _$LightingEnumMap[instance.lighting]!,
  'complexity': instance.complexity,
  'colorDistribution': instance.colorDistribution.toJson(),
  'resolution': _$ResolutionEnumMap[instance.resolution]!,
  'noiseLevel': _$NoiseLevelEnumMap[instance.noiseLevel]!,
};

const _$HazeLevelEnumMap = {
  HazeLevel.light: 'light',
  HazeLevel.moderate: 'moderate',
  HazeLevel.heavy: 'heavy',
};

const _$SceneTypeEnumMap = {
  SceneType.urban: 'urban',
  SceneType.landscape: 'landscape',
  SceneType.building: 'building',
  SceneType.night: 'night',
  SceneType.backlight: 'backlight',
  SceneType.indoor: 'indoor',
};

const _$LightingEnumMap = {
  Lighting.bright: 'bright',
  Lighting.normal: 'normal',
  Lighting.dark: 'dark',
  Lighting.veryDark: 'veryDark',
  Lighting.backlight: 'backlight',
};

const _$ResolutionEnumMap = {
  Resolution.sd: 'sd',
  Resolution.hd: 'hd',
  Resolution.uhd: 'uhd',
};

const _$NoiseLevelEnumMap = {
  NoiseLevel.low: 'low',
  NoiseLevel.medium: 'medium',
  NoiseLevel.high: 'high',
};

RecommendedAlgorithm _$RecommendedAlgorithmFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendedAlgorithm', json, ($checkedConvert) {
  final val = RecommendedAlgorithm(
    recommendationId: $checkedConvert(
      'recommendationId',
      (v) => (v as num?)?.toInt(),
    ),
    algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
    algorithmName: $checkedConvert('algorithmName', (v) => v as String),
    matchScore: $checkedConvert('matchScore', (v) => (v as num).toInt()),
    reason: $checkedConvert('reason', (v) => v as String),
    rating: $checkedConvert('rating', (v) => (v as num).toInt()),
    estimatedTime: $checkedConvert(
      'estimatedTime',
      (v) => (v as num?)?.toInt(),
    ),
    effectDescription: $checkedConvert(
      'effectDescription',
      (v) => v as String?,
    ),
  );
  return val;
});

Map<String, dynamic> _$RecommendedAlgorithmToJson(
  RecommendedAlgorithm instance,
) => <String, dynamic>{
  if (instance.recommendationId case final value?) 'recommendationId': value,
  'algorithmId': instance.algorithmId,
  'algorithmName': instance.algorithmName,
  'matchScore': instance.matchScore,
  'reason': instance.reason,
  'rating': instance.rating,
  if (instance.estimatedTime case final value?) 'estimatedTime': value,
  if (instance.effectDescription case final value?) 'effectDescription': value,
};

RecommendationResult _$RecommendationResultFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendationResult', json, ($checkedConvert) {
  final val = RecommendationResult(
    analysis: $checkedConvert(
      'analysis',
      (v) => ImageFeatureAnalysis.fromJson(v as Map<String, dynamic>),
    ),
    recommendations: $checkedConvert(
      'recommendations',
      (v) => (v as List<dynamic>)
          .map((e) => RecommendedAlgorithm.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$RecommendationResultToJson(
  RecommendationResult instance,
) => <String, dynamic>{
  'analysis': instance.analysis.toJson(),
  'recommendations': instance.recommendations.map((e) => e.toJson()).toList(),
};

RecommendationFeedback _$RecommendationFeedbackFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendationFeedback', json, ($checkedConvert) {
  final val = RecommendationFeedback(
    recommendationId: $checkedConvert(
      'recommendationId',
      (v) => (v as num).toInt(),
    ),
    useful: $checkedConvert('useful', (v) => v as bool),
  );
  return val;
});

Map<String, dynamic> _$RecommendationFeedbackToJson(
  RecommendationFeedback instance,
) => <String, dynamic>{
  'recommendationId': instance.recommendationId,
  'useful': instance.useful,
};

RecommendationRule _$RecommendationRuleFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RecommendationRule', json, ($checkedConvert) {
      final val = RecommendationRule(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        ruleName: $checkedConvert('ruleName', (v) => v as String),
        sceneType: $checkedConvert('sceneType', (v) => v as String),
        algorithmIds: $checkedConvert(
          'algorithmIds',
          (v) => (v as List<dynamic>).map((e) => (e as num).toInt()).toList(),
        ),
        weight: $checkedConvert('weight', (v) => (v as num).toInt()),
        enabled: $checkedConvert('enabled', (v) => v as bool),
      );
      return val;
    });

Map<String, dynamic> _$RecommendationRuleToJson(RecommendationRule instance) =>
    <String, dynamic>{
      if (instance.id case final value?) 'id': value,
      'ruleName': instance.ruleName,
      'sceneType': instance.sceneType,
      'algorithmIds': instance.algorithmIds,
      'weight': instance.weight,
      'enabled': instance.enabled,
    };

RecommendationTrendItem _$RecommendationTrendItemFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendationTrendItem', json, ($checkedConvert) {
  final val = RecommendationTrendItem(
    date: $checkedConvert('date', (v) => v as String),
    adoptionRate: $checkedConvert('adoptionRate', (v) => (v as num).toDouble()),
  );
  return val;
});

Map<String, dynamic> _$RecommendationTrendItemToJson(
  RecommendationTrendItem instance,
) => <String, dynamic>{
  'date': instance.date,
  'adoptionRate': instance.adoptionRate,
};

RecommendationReport _$RecommendationReportFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendationReport', json, ($checkedConvert) {
  final val = RecommendationReport(
    totalRecommendations: $checkedConvert(
      'totalRecommendations',
      (v) => (v as num).toInt(),
    ),
    adoptionRate: $checkedConvert('adoptionRate', (v) => (v as num).toDouble()),
    satisfactionRate: $checkedConvert(
      'satisfactionRate',
      (v) => (v as num).toDouble(),
    ),
    coverageRate: $checkedConvert('coverageRate', (v) => (v as num).toDouble()),
    coldStartSuccessRate: $checkedConvert(
      'coldStartSuccessRate',
      (v) => (v as num).toDouble(),
    ),
    trend: $checkedConvert(
      'trend',
      (v) => (v as List<dynamic>)
          .map(
            (e) => RecommendationTrendItem.fromJson(e as Map<String, dynamic>),
          )
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$RecommendationReportToJson(
  RecommendationReport instance,
) => <String, dynamic>{
  'totalRecommendations': instance.totalRecommendations,
  'adoptionRate': instance.adoptionRate,
  'satisfactionRate': instance.satisfactionRate,
  'coverageRate': instance.coverageRate,
  'coldStartSuccessRate': instance.coldStartSuccessRate,
  'trend': instance.trend.map((e) => e.toJson()).toList(),
};
