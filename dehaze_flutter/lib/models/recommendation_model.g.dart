// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'recommendation_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AnalyzeForm _$AnalyzeFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnalyzeForm', json, ($checkedConvert) {
      final val = AnalyzeForm(
        imageUrl: $checkedConvert('imageUrl', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AnalyzeFormToJson(AnalyzeForm instance) =>
    <String, dynamic>{
      if (instance.imageUrl case final value?) 'imageUrl': value,
    };

ImageFeatureAnalysisVO _$ImageFeatureAnalysisVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageFeatureAnalysisVO', json, ($checkedConvert) {
  final val = ImageFeatureAnalysisVO(
    imageMd5: $checkedConvert('imageMd5', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$ImageFeatureAnalysisVOToJson(
  ImageFeatureAnalysisVO instance,
) => <String, dynamic>{
  if (instance.imageMd5 case final value?) 'imageMd5': value,
};

RecommendedAlgorithmVO _$RecommendedAlgorithmVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RecommendedAlgorithmVO', json, ($checkedConvert) {
  final val = RecommendedAlgorithmVO(
    algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
    algorithmName: $checkedConvert('algorithmName', (v) => v as String),
    matchScore: $checkedConvert('matchScore', (v) => (v as num?)?.toInt()),
    reason: $checkedConvert('reason', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$RecommendedAlgorithmVOToJson(
  RecommendedAlgorithmVO instance,
) => <String, dynamic>{
  'algorithmId': instance.algorithmId,
  'algorithmName': instance.algorithmName,
  if (instance.matchScore case final value?) 'matchScore': value,
  if (instance.reason case final value?) 'reason': value,
};
