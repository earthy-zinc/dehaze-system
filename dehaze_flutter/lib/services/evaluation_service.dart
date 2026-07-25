import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/evaluation_model.dart';
import '../models/prediction_model.dart';
import 'prediction_service.dart';

/// 评估服务
///
/// 封装效果评估相关 API：
/// - evaluate: POST 提交评估任务，立即返回 logId + status
/// - getEvalTaskStatus: GET 查询任务状态
/// - evaluateAndWait: 提交并轮询至终态
class EvaluationService {
  const EvaluationService(this._dio);

  final Dio _dio;

  /// 提交效果评估任务
  ///
  /// POST /evaluation
  Future<EvaluationResult> evaluate(EvaluationRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.evaluation,
      data: request.toJson(),
    );
    return EvaluationResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询评估任务状态
  ///
  /// GET /evaluation/{taskId}
  Future<EvaluationResult> getEvalTaskStatus(int taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.evaluation}/$taskId',
    );
    return EvaluationResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 提交评估并等待结果（POST + 轮询 GET）
  ///
  /// - POST 立即返回，若 status=completed 直接返回
  /// - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
  Future<EvaluationResult> evaluateAndWait(
    EvaluationRequest request, {
    PollOptions? options,
  }) async {
    final result = await evaluate(request);
    if (result.status != TaskStatus.processing) {
      return result;
    }
    return _pollEvalTask(result.logId, options);
  }

  Future<EvaluationResult> _pollEvalTask(
    int logId,
    PollOptions? options,
  ) async {
    final interval = options?.intervalMs ?? 2000;
    final timeout = options?.timeoutMs ?? 120000;
    final deadline = DateTime.now().add(Duration(milliseconds: timeout));

    while (DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(Duration(milliseconds: interval));
      final result = await getEvalTaskStatus(logId);
      options?.onPoll?.call(result.status);
      if (result.status == TaskStatus.completed ||
          result.status == TaskStatus.failed) {
        return result;
      }
    }
    throw Exception('评估任务 $logId 超时（${timeout}ms）');
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
