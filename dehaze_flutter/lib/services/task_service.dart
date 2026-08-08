import 'dart:convert';

import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../core/network/page_result.dart';
import '../models/task_model.dart';

/// 任务服务
///
/// 对齐 JS SDK TaskAPI，封装统一任务相关 API：
/// - 用户端：create / getMyTasks / getById / getStatus / cancel / retry / download
/// - 管理端：getPage（分页查询）
class TaskService {
  const TaskService(this._dio);

  final Dio _dio;

  // ==================== 用户端 API ====================

  /// 创建任务
  ///
  /// POST /api/v1/tasks
  ///
  /// JS SDK 将 type 外的字段打包为 paramsJson 发送，此处保持一致。
  Future<TaskVO> create(TaskCreateForm form) async {
    final Map<String, dynamic> data = <String, dynamic>{'type': form.type};
    final extras = <String, dynamic>{};
    if (form.targetId != null) extras['targetId'] = form.targetId;
    if (form.targetIds != null && form.targetIds!.isNotEmpty) {
      extras['targetIds'] = form.targetIds;
    }
    if (form.options != null && form.options!.isNotEmpty) {
      extras['options'] = form.options;
    }
    if (extras.isNotEmpty) {
      data['paramsJson'] = jsonEncode(extras);
    }

    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.tasks,
      data: data,
    );
    return TaskVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询我的任务列表
  ///
  /// GET /api/v1/tasks（后端按当前用户过滤）
  Future<PageResult<TaskVO>> getMyTasks({
    int pageNum = 1,
    int pageSize = 10,
    TaskStatusType? status,
    String? taskType,
    TaskCategory? taskCategory,
  }) async {
    final query = TaskQuery(
      pageNum: pageNum,
      pageSize: pageSize,
      status: status,
      taskType: taskType,
      taskCategory: taskCategory,
    );
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.tasks,
      queryParameters: query.toQueryParameters(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => TaskVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }

  /// 查询任务详情/状态
  ///
  /// GET /api/v1/tasks/{id}
  Future<TaskVO> getById(String taskId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '${ApiConstants.tasks}/$taskId',
    );
    return TaskVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 查询任务状态（与 getById 同路由，语义别名）
  ///
  /// GET /api/v1/tasks/{id}
  Future<TaskVO> getStatus(String taskId) async {
    return getById(taskId);
  }

  /// 取消任务
  ///
  /// POST /api/v1/tasks/{id}/cancel
  Future<void> cancel(String taskId) async {
    await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.tasks}/$taskId/cancel',
    );
  }

  /// 重试失败的任务
  ///
  /// POST /api/v1/tasks/{id}/retry
  Future<TaskVO> retry(String taskId) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '${ApiConstants.tasks}/$taskId/retry',
    );
    return TaskVO.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 下载任务结果（返回 Response 流）
  ///
  /// GET /api/v1/tasks/{id}/download
  ///
  /// 后端返回 302 重定向，Dio 默认跟随重定向，返回最终文件流。
  /// 调用方通过 response.data 读取 `Stream<List<int>>`。
  Future<Response<dynamic>> download(String taskId) async {
    return _dio.get<dynamic>(
      '${ApiConstants.tasks}/$taskId/download',
      options: Options(responseType: ResponseType.stream),
    );
  }

  // ==================== 管理端 API ====================

  /// 管理端分页查询任务列表
  ///
  /// GET /api/v1/tasks（通过 TaskQuery 筛选）
  Future<PageResult<TaskVO>> getPage(TaskQuery query) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.tasks,
      queryParameters: query.toQueryParameters(),
    );
    final data = response.data!['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>? ?? [])
        .map((e) => TaskVO.fromJson(e as Map<String, dynamic>))
        .toList();
    final total = data['total'] as int? ?? 0;
    return PageResult(list: list, total: total);
  }
}
