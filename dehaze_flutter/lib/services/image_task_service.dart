import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../core/network/task_poller.dart';
import '../models/image_task_model.dart';

/// 图像任务服务
///
/// 封装统一图像任务 API（提交-轮询-取消-历史）：
/// - submit: POST /images/edits，立即返回 taskId + status
/// - getTaskStatus: GET /images/tasks/{taskId}
/// - cancelTask: POST /images/tasks/{taskId}/cancel
/// - submitAndWait: 提交并轮询至终态
class ImageTaskService {
  const ImageTaskService(this._dio);

  final Dio _dio;

  /// 提交图像编辑任务
  ///
  /// POST /images/edits
  Future<ImageTaskResponse> submit(ImageEditRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.imagesEdits,
      data: request.toJson(),
    );
    return ImageTaskResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询图像任务状态
  ///
  /// GET /images/tasks/{taskId}
  Future<ImageTaskResponse> getTaskStatus(int taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.imagesTasks}/$taskId',
    );
    return ImageTaskResponse.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 取消图像任务（幂等，仅 processing 可取消）
  ///
  /// POST /images/tasks/{taskId}/cancel
  Future<void> cancelTask(int taskId) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.imagesTasks}/$taskId/cancel',
    );
  }

  /// 提交并等待结果（POST + 轮询 GET）
  ///
  /// - POST 立即返回，若 status 非 processing 直接返回（缓存命中即 completed）
  /// - status=processing 时按 intervalMs 轮询 GET，直到终态、取消或超时
  /// - [onSubmitted] 用于让调用方拿到 taskId（取消任务时需要）
  Future<ImageTaskResponse> submitAndWait(
    ImageEditRequest request, {
    PollOptions? options,
    void Function(int taskId)? onSubmitted,
  }) async {
    final result = await submit(request);
    if (result.status != TaskStatus.processing) {
      return result;
    }
    final taskId = result.taskId!;
    onSubmitted?.call(taskId);
    return waitForTask(taskId, options: options);
  }

  /// 轮询指定任务直到终态、取消或超时
  Future<ImageTaskResponse> waitForTask(int taskId, {PollOptions? options}) {
    return pollTask(
      getTaskStatus,
      taskId,
      statusOf: (r) => r.status,
      options: options,
    );
  }

  /// 获取图像任务历史
  ///
  /// GET /images/tasks
  Future<PageResult<ImageTaskVO>> getImageTasks({
    int pageNum = 1,
    int pageSize = 10,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.imagesTasks,
      queryParameters: {'pageNum': pageNum, 'pageSize': pageSize},
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => ImageTaskVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
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
    return PresetVO.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 更新自定义预设
  ///
  /// PUT /presets/{id}
  Future<PresetVO> updatePreset(int id, PresetForm data) async {
    final response = await _dio.put<Map<String, dynamic>>(
      '${ApiConstants.presets}/$id',
      data: data.toJson(),
    );
    return PresetVO.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 删除自定义预设
  ///
  /// DELETE /presets/{id}
  Future<void> deletePreset(int id) async {
    await _dio.delete<Map<String, dynamic>>('${ApiConstants.presets}/$id');
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
