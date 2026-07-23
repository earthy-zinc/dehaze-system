import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/evaluation_model.dart';

/// 评估服务
///
/// 封装效果评估相关 API：
/// - evaluate: 执行效果评估（同步返回指标）
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
    // ResponseInterceptor 已保证 code=='00000'，失败已 reject 为 ApiException
    return EvaluationResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
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
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => EvaluationResult.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }
}
