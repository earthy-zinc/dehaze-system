import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/recommendation_model.dart';

/// 推荐服务
///
/// 根据图像特征、用户偏好、算法表现等多维信息，为用户推荐最合适的去雾算法。
///
/// - analyze / getRecommendations / submitFeedback：仅需登录用户身份
/// - getRules / updateRule / getReport：管理员接口
class RecommendationService {
  const RecommendationService(this._dio);

  final Dio _dio;

  /// 图像特征分析
  ///
  /// POST /recommendations/analyze
  /// 上传图片或指定 imageId，返回 7 维特征分析结果。
  Future<ImageFeatureAnalysis> analyze(AnalyzeRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.recommendationsAnalyze,
      data: request.toJson(),
    );
    return ImageFeatureAnalysis.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 获取算法推荐
  ///
  /// GET /recommendations/algorithms
  /// 基于分析结果返回 Top 3 推荐算法及匹配度和理由。
  Future<List<RecommendedAlgorithm>> getRecommendations({
    int? analysisId,
    String? imageMd5,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.recommendationsAlgorithms,
      queryParameters: {
        if (analysisId != null) 'analysisId': analysisId,
        if (imageMd5 != null) 'imageMd5': imageMd5,
      },
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => RecommendedAlgorithm.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 提交推荐反馈
  ///
  /// POST /recommendations/feedback
  /// 用户对推荐结果进行有用/无用反馈，反馈数据用于优化推荐模型。
  Future<int> submitFeedback(RecommendationFeedback feedback) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.recommendationsFeedback,
      data: feedback.toJson(),
    );
    return (response.data!['data']['id'] as num).toInt();
  }

  /// 获取推荐规则配置（管理员）
  ///
  /// GET /recommendations/rules
  Future<List<RecommendationRule>> getRules() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.recommendationsRules,
    );
    final list = response.data!['data'] as List<dynamic>? ?? [];
    return list
        .map((e) => RecommendationRule.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// 更新推荐规则配置（管理员）
  ///
  /// PUT /recommendations/rules
  Future<int> updateRule(int id, RecommendationRule rule) async {
    final response = await _dio.put<Map<String, dynamic>>(
      ApiConstants.recommendationsRules,
      queryParameters: {'id': id},
      data: rule.toJson(),
    );
    return (response.data!['data'] as num).toInt();
  }

  /// 推荐效果报表（管理员）
  ///
  /// GET /recommendations/report
  Future<RecommendationReport> getReport({
    String? startDate,
    String? endDate,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.recommendationsReport,
      queryParameters: {
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      },
    );
    return RecommendationReport.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
