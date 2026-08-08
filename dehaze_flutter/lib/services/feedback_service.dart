import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/feedback_model.dart';

/// 反馈与评分服务
///
/// 封装评分（Rating）和反馈（Feedback）相关 API，对齐 JS SDK FeedbackAPI。
class FeedbackService {
  const FeedbackService(this._dio);

  final Dio _dio;

  // ==================== 评分 ====================

  /// 提交评分
  ///
  /// POST /api/v1/feedback/ratings
  Future<void> createRating(RatingCreateForm data) async {
    await _dio.post<Map<String, dynamic>>(
      ApiConstants.feedbackRatings,
      data: data.toJson(),
    );
  }

  /// 获取我的评分列表
  ///
  /// GET /api/v1/feedback/ratings/my
  Future<PageResult<MyRatingVO>> getMyRatings({
    int pageNum = 1,
    int pageSize = 10,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.feedbackRatingsMy,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
      },
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => MyRatingVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取评分分页列表（后台）
  ///
  /// GET /api/v1/feedback/ratings/page
  Future<PageResult<RatingPageVO>> getRatingPage(RatingQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.feedbackRatingsPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => RatingPageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取评分详情
  ///
  /// GET /api/v1/feedback/ratings/{id}
  Future<RatingDetailVO> getRatingDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.feedbackRatings}/$id',
    );
    return RatingDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 按预测日志查询评分
  ///
  /// GET /api/v1/feedback/ratings/by-prediction/{predictionLogId}
  Future<MyRatingVO?> getRatingByPrediction(int predictionLogId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.feedbackRatingsByPrediction}/$predictionLogId',
    );
    final data = response.data!['data'];
    if (data == null) return null;
    return MyRatingVO.fromJson(data as Map<String, dynamic>);
  }

  /// 获取评分统计
  ///
  /// GET /api/v1/feedback/ratings/stats/{algorithmId}
  Future<RatingStatsVO> getRatingStats(int algorithmId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.feedbackRatingsStats}/$algorithmId',
    );
    return RatingStatsVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ==================== 反馈 ====================

  /// 提交反馈
  ///
  /// POST /api/v1/feedback
  /// 返回反馈 ID
  Future<int> createFeedback(FeedbackCreateForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.feedback,
      data: data.toJson(),
    );
    final result = response.data!['data'] as Map<String, dynamic>;
    return result['id'] as int;
  }

  /// 获取我的反馈列表
  ///
  /// GET /api/v1/feedback/my
  Future<PageResult<FeedbackPageVO>> getMyFeedbacks({
    int pageNum = 1,
    int pageSize = 10,
    FeedbackType? type,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.feedbackMy,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (type != null) 'type': type.value,
      },
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => FeedbackPageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取反馈详情
  ///
  /// GET /api/v1/feedback/{id}
  Future<FeedbackDetailVO> getFeedbackDetail(int id) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.feedback}/$id',
    );
    return FeedbackDetailVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 补充反馈
  ///
  /// POST /api/v1/feedback/supplement
  Future<void> supplementFeedback(FeedbackSupplementForm data) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.feedback}/supplement',
      data: data.toJson(),
    );
  }

  /// 获取反馈分页列表（后台）
  ///
  /// GET /api/v1/feedback/page
  Future<PageResult<FeedbackPageVO>> getFeedbackPage(FeedbackQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.feedbackPage,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => FeedbackPageVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 获取反馈统计
  ///
  /// GET /api/v1/feedback/stats
  Future<FeedbackStatsVO> getFeedbackStats() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.feedbackStats,
    );
    return FeedbackStatsVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 回复反馈
  ///
  /// POST /api/v1/feedback/reply
  Future<void> replyFeedback(FeedbackReplyForm data) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.feedback}/reply',
      data: data.toJson(),
    );
  }

  /// 分配反馈
  ///
  /// PUT /api/v1/feedback/assign
  Future<void> assignFeedback(FeedbackAssignForm data) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.feedback}/assign',
      data: data.toJson(),
    );
  }

  /// 关闭反馈
  ///
  /// PUT /api/v1/feedback/close
  Future<void> closeFeedback(FeedbackCloseForm data) async {
    await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.feedback}/close',
      data: data.toJson(),
    );
  }
}
