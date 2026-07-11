import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/evaluation_model.dart';

/// 评估服务
///
/// 封装效果评估相关 API：
/// - evaluate: 执行效果评估
/// - getEvaluationStatus: 查询评估任务状态
/// - getEvaluationLogs: 获取评估日志列表
class EvaluationService {
  const EvaluationService(this._dio);

  final Dio _dio;

  /// 执行效果评估
  ///
  /// POST /evaluation
  Future<EvaluationResult> evaluate(EvaluationRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.evaluation,
      data: request.toJson(),
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return EvaluationResult.fromJson(
        result['data'] as Map<String, dynamic>,
      );
    }
    throw Exception(result['msg'] ?? '评估请求失败');
  }

  /// 查询评估任务状态
  ///
  /// GET /evaluation/{taskId}
  Future<EvaluationResult> getEvaluationStatus(String taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.evaluation}/$taskId',
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return EvaluationResult.fromJson(
        result['data'] as Map<String, dynamic>,
      );
    }
    throw Exception(result['msg'] ?? '查询评估状态失败');
  }

  /// 获取评估日志列表
  ///
  /// GET /evaluation/logs
  Future<PageResult<EvaluationResult>> getEvaluationLogs({
    int pageNum = 1,
    int pageSize = 10,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.evaluationLogs,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
      },
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      final data = result['data'] as Map<String, dynamic>;
      final list = (data['list'] as List<dynamic>? ?? [])
          .map((e) => EvaluationResult.fromJson(e as Map<String, dynamic>))
          .toList();
      final total = data['total'] as int? ?? 0;
      return PageResult(list: list, total: total);
    }
    throw Exception(result['msg'] ?? '获取评估日志失败');
  }
}
