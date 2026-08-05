import 'package:json_annotation/json_annotation.dart';

part 'recommendation_model.g.dart';

/// 图像分析请求（对应后端 AnalyzeForm）
@JsonSerializable()
class AnalyzeForm {
  const AnalyzeForm({this.imageUrl});

  factory AnalyzeForm.fromJson(Map<String, dynamic> json) =>
      _$AnalyzeFormFromJson(json);

  /// 图片 URL（与 imageId 二选一，移动端统一使用 URL）
  @JsonKey(name: 'imageUrl')
  final String? imageUrl;

  Map<String, dynamic> toJson() => _$AnalyzeFormToJson(this);
}

/// 图像特征分析结果（对应后端 ImageFeatureAnalysisVO）
@JsonSerializable()
class ImageFeatureAnalysisVO {
  const ImageFeatureAnalysisVO({this.imageMd5});

  factory ImageFeatureAnalysisVO.fromJson(Map<String, dynamic> json) =>
      _$ImageFeatureAnalysisVOFromJson(json);

  /// 图像 MD5 值，用于关联推荐查询
  @JsonKey(name: 'imageMd5')
  final String? imageMd5;

  Map<String, dynamic> toJson() => _$ImageFeatureAnalysisVOToJson(this);
}

/// 推荐算法项（对应后端 RecommendedAlgorithmVO）
@JsonSerializable()
class RecommendedAlgorithmVO {
  const RecommendedAlgorithmVO({
    required this.algorithmId,
    required this.algorithmName,
    this.matchScore,
    this.reason,
  });

  factory RecommendedAlgorithmVO.fromJson(Map<String, dynamic> json) =>
      _$RecommendedAlgorithmVOFromJson(json);

  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  @JsonKey(name: 'algorithmName')
  final String algorithmName;

  /// 匹配度(0-100)
  @JsonKey(name: 'matchScore')
  final int? matchScore;

  /// 推荐理由
  final String? reason;

  Map<String, dynamic> toJson() => _$RecommendedAlgorithmVOToJson(this);
}
