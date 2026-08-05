import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/recommendation_model.dart';

/// 推荐服务
///
/// 封装智能推荐两步流程：
/// - analyze: POST /recommendations/analyze 图像特征分析，返回 imageMd5
/// - getRecommendations: GET /recommendations/algorithms 基于 imageMd5 获取推荐算法列表
class RecommendationService {
  const RecommendationService(this._dio);

  final Dio _dio;

  /// 图像特征分析
  Future<ImageFeatureAnalysisVO> analyze(AnalyzeForm form) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.recommendationsAnalyze,
      data: form.toJson(),
    );
    return ImageFeatureAnalysisVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取算法推荐
  Future<List<RecommendedAlgorithmVO>> getRecommendations(
    String imageMd5,
  ) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.recommendationsAlgorithms,
      queryParameters: {'imageMd5': imageMd5},
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => RecommendedAlgorithmVO.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}
