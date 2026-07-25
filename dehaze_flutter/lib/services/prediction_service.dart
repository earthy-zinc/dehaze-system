import 'dart:async';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/prediction_model.dart';

/// 轮询配置
class PollOptions {
  const PollOptions({
    this.intervalMs = 2000,
    this.timeoutMs = 120000,
    this.onPoll,
  });

  /// 轮询间隔（毫秒）
  final int intervalMs;

  /// 最大等待时间（毫秒）
  final int timeoutMs;

  /// 每次轮询回调
  final void Function(TaskStatus status)? onPoll;
}

/// 预测服务
///
/// 封装模型预测相关 API：
/// - predict: POST 提交预测任务，立即返回 logId + status
/// - getPredTaskStatus: GET 查询任务状态
/// - predictAndWait: 提交并轮询至终态
class PredictionService {
  const PredictionService(this._dio);

  final Dio _dio;

  /// 提交模型预测任务
  ///
  /// POST /prediction
  Future<PredictionResponse> predict(PredictionRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.prediction,
      data: request.toJson(),
    );
    return PredictionResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询预测任务状态
  ///
  /// GET /prediction/{taskId}
  Future<PredictionResponse> getPredTaskStatus(int taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.prediction}/$taskId',
    );
    return PredictionResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 提交预测并等待结果（POST + 轮询 GET）
  ///
  /// - POST 立即返回，若 status=completed 直接返回
  /// - status=processing 时按 intervalMs 轮询 GET，直到 completed/failed 或超时
  Future<PredictionResponse> predictAndWait(
    PredictionRequest request, {
    PollOptions? options,
  }) async {
    final result = await predict(request);
    if (result.status != TaskStatus.processing) {
      return result;
    }
    return _pollPredTask(result.logId, options);
  }

  Future<PredictionResponse> _pollPredTask(
    int logId,
    PollOptions? options,
  ) async {
    final interval = options?.intervalMs ?? 2000;
    final timeout = options?.timeoutMs ?? 120000;
    final deadline = DateTime.now().add(Duration(milliseconds: timeout));

    while (DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(Duration(milliseconds: interval));
      final result = await getPredTaskStatus(logId);
      options?.onPoll?.call(result.status);
      if (result.status == TaskStatus.completed ||
          result.status == TaskStatus.failed) {
        return result;
      }
    }
    throw Exception('预测任务 $logId 超时（${timeout}ms）');
  }

  /// 获取预测日志列表
  ///
  /// GET /prediction/logs
  Future<PageResult<PredictionLog>> getPredictionLogs({
    int pageNum = 1,
    int pageSize = 10,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.predictionLogs,
      queryParameters: {
        'pageNum': pageNum,
        'pageSize': pageSize,
      },
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => PredictionLog.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }
}
