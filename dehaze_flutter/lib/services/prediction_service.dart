import 'dart:async';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../core/network/task_poller.dart';
import '../models/prediction_model.dart';

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
    return pollTask(
      getPredTaskStatus,
      result.logId!,
      statusOf: (r) => r.status,
      options: options,
    );
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

  // ===== 批量预测 =====

  /// 批量预测（一次提交多张图片，最多 20 张）
  ///
  /// POST /prediction/batch
  Future<BatchPredictionResult> batchPredict(BatchPredictionForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.predictionBatch,
      data: data.toJson(),
    );
    return BatchPredictionResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ===== VIP 配额 =====

  /// 查询 VIP 配额（剩余处理次数）
  ///
  /// GET /prediction/quota
  Future<PredictionQuota> getQuota() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.predictionQuota,
    );
    return PredictionQuota.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  // ===== 参数预设 =====

  /// 获取参数预设分页列表（系统预设 + 用户自定义）
  ///
  /// GET /presets
  Future<PageResult<PresetVO>> getPresets(PresetQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.presets,
      queryParameters: query.toQuery(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => PresetVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 创建自定义预设
  ///
  /// POST /presets
  Future<PresetVO> createPreset(PresetForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.presets,
      data: data.toJson(),
    );
    return PresetVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 更新自定义预设
  ///
  /// PUT /presets/{id}
  Future<PresetVO> updatePreset(int id, PresetForm data) async {
    final response = await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.presets}/$id',
      data: data.toJson(),
    );
    return PresetVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 删除自定义预设
  ///
  /// DELETE /presets/{id}
  Future<void> deletePreset(int id) async {
    await _dio.delete<Map<String, dynamic>>(
      '${ApiConstants.presets}/$id',
    );
  }

  // ===== 对比报告 =====

  /// 生成对比报告（异步任务，通过任务管理追踪进度）
  ///
  /// POST /compare/report
  Future<CompareReportResult> generateReport(CompareReportForm data) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.compareReport,
      data: data.toJson(),
    );
    return CompareReportResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询对比报告任务状态（报告生成完成后返回下载URL）
  ///
  /// GET /compare/report/{taskId}
  Future<CompareReportResult> getReportStatus(int taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.compareReport}/$taskId',
    );
    return CompareReportResult.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
