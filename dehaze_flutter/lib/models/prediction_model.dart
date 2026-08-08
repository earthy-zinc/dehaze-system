import 'dart:convert';

import 'package:json_annotation/json_annotation.dart';

part 'prediction_model.g.dart';

/// 任务状态（后端以整数序列化）
enum TaskStatus {
  processing,
  completed,
  failed;

  static TaskStatus fromValue(int? value) {
    switch (value) {
      case 2:
        return TaskStatus.completed;
      case 3:
        return TaskStatus.failed;
      default:
        return TaskStatus.processing;
    }
  }
}

TaskStatus _statusFromJson(int? value) => TaskStatus.fromValue(value);

int? _statusToJson(TaskStatus? status) {
  if (status == null) return null;
  return switch (status) {
    TaskStatus.completed => 2,
    TaskStatus.failed => 3,
    TaskStatus.processing => 1,
  };
}

/// 预测请求
///
/// 对应后端 PredictionForm：
/// algorithmId（Long，必填）、fileId（Long）、imageUrl（String）、params（String，JSON）
@JsonSerializable()
class PredictionRequest {
  const PredictionRequest({
    required this.algorithmId,
    this.fileId,
    this.imageUrl,
    this.params,
    this.recommendedBy,
  });

  factory PredictionRequest.fromJson(Map<String, dynamic> json) =>
      _$PredictionRequestFromJson(json);

  /// 算法 ID
  final int algorithmId;

  /// 原始图片文件 ID
  final int? fileId;

  /// 原始图片 URL（fileId 与 imageUrl 二选一）
  final String? imageUrl;

  /// 算法参数
  ///
  /// 后端 `params` 字段为 String 类型，需序列化为 JSON 字符串传输。
  @JsonKey(toJson: _paramsToJson, fromJson: _paramsFromJson)
  final Map<String, dynamic>? params;

  /// 推荐算法 ID（来自推荐管理模块，用于追踪推荐采纳率）
  final int? recommendedBy;

  static String? _paramsToJson(Map<String, dynamic>? params) =>
      params == null ? null : jsonEncode(params);

  static Map<String, dynamic>? _paramsFromJson(String? params) =>
      params == null || params.isEmpty
          ? null
          : jsonDecode(params) as Map<String, dynamic>;

  Map<String, dynamic> toJson() => _$PredictionRequestToJson(this);
}

/// 预测响应
///
/// 对应后端 PredictionResultVO：异步任务模式。
/// POST 立即返回 logId + status=processing；GET 轮询至 completed/failed 时返回完整字段。
@JsonSerializable()
class PredictionResponse {
  const PredictionResponse({
    this.logId,
    required this.status,
    this.resultUrl,
    this.resultThumbnailUrl,
    this.time,
    this.errorMessage,
    this.fromCache,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) =>
      _$PredictionResponseFromJson(json);

  /// 预测日志 ID（POST 返回时可能为 null）
  final int? logId;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus status;

  /// 结果图片 URL（status=completed 时返回）
  final String? resultUrl;

  /// 结果缩略图 URL（status=completed 时返回）
  final String? resultThumbnailUrl;

  /// 处理耗时（毫秒）
  final int? time;

  /// 失败错误信息（status=failed 时返回）
  final String? errorMessage;

  /// POST 缓存命中时返回，GET 不返回
  final bool? fromCache;

  Map<String, dynamic> toJson() => _$PredictionResponseToJson(this);

  /// 是否已拿到处理结果
  bool get hasResult =>
      status == TaskStatus.completed &&
      resultUrl != null &&
      resultUrl!.isNotEmpty;
}

/// 预测日志
///
/// 对应后端 PredLogVO。
@JsonSerializable()
class PredictionLog {
  const PredictionLog({
    required this.id,
    required this.algorithmName,
    required this.createTime,
    this.algorithmId,
    this.originUrl,
    this.predUrl,
    this.status,
    this.errorMessage,
    this.time,
  });

  factory PredictionLog.fromJson(Map<String, dynamic> json) =>
      _$PredictionLogFromJson(json);

  final int id;

  final int? algorithmId;

  final String algorithmName;

  /// 原始图片 URL
  final String? originUrl;

  /// 预测结果图片 URL（失败的记录可能为空）
  final String? predUrl;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus? status;

  /// 失败错误信息
  final String? errorMessage;

  /// 处理耗时（毫秒）
  final int? time;

  /// 创建时间
  final String createTime;

  Map<String, dynamic> toJson() => _$PredictionLogToJson(this);
}

// ===== 批量预测 =====

/// 批量预测请求项
class BatchPredictionItem {
  final int? fileId;
  final String? imageUrl;
  final String? params;

  const BatchPredictionItem({this.fileId, this.imageUrl, this.params});

  factory BatchPredictionItem.fromJson(Map<String, dynamic> json) {
    return BatchPredictionItem(
      fileId: (json['file_id'] ?? json['fileId']) as int?,
      imageUrl: (json['image_url'] ?? json['imageUrl']) as String?,
      params: (json['params']) as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        if (fileId != null) 'file_id': fileId,
        if (imageUrl != null) 'image_url': imageUrl,
        if (params != null) 'params': params,
      };
}

/// 批量预测请求
class BatchPredictionForm {
  final int algorithmId;
  final List<BatchPredictionItem> items;
  final int? recommendedBy;

  const BatchPredictionForm({
    required this.algorithmId,
    required this.items,
    this.recommendedBy,
  });

  Map<String, dynamic> toJson() => {
        'algorithm_id': algorithmId,
        'items': items.map((e) => e.toJson()).toList(),
        if (recommendedBy != null) 'recommended_by': recommendedBy,
      };
}

/// 批量预测结果
class BatchPredictionResult {
  final int total;
  final List<PredictionResponse> results;

  const BatchPredictionResult({required this.total, required this.results});

  factory BatchPredictionResult.fromJson(Map<String, dynamic> json) {
    return BatchPredictionResult(
      total: (json['total'] as num).toInt(),
      results: (json['results'] as List<dynamic>)
          .map((e) => PredictionResponse.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() => {
        'total': total,
        'results': results.map((e) => e.toJson()).toList(),
      };
}

// ===== VIP 配额 =====

/// VIP 配额
class PredictionQuota {
  final int remaining;
  final int total;
  final int used;
  final String resetDate;

  const PredictionQuota({
    required this.remaining,
    required this.total,
    required this.used,
    required this.resetDate,
  });

  factory PredictionQuota.fromJson(Map<String, dynamic> json) {
    return PredictionQuota(
      remaining: (json['remaining'] as num).toInt(),
      total: (json['total'] as num).toInt(),
      used: (json['used'] as num).toInt(),
      resetDate: json['reset_date'] as String? ?? json['resetDate'] as String,
    );
  }

  Map<String, dynamic> toJson() => {
        'remaining': remaining,
        'total': total,
        'used': used,
        'reset_date': resetDate,
      };
}

// ===== 参数预设 =====

/// 参数预设表单
class PresetForm {
  final int? id;
  final String name;
  final int algorithmId;
  final String params;
  final bool? isSystem;

  const PresetForm({
    this.id,
    required this.name,
    required this.algorithmId,
    required this.params,
    this.isSystem,
  });

  factory PresetForm.fromJson(Map<String, dynamic> json) {
    return PresetForm(
      id: (json['id'] as num?)?.toInt(),
      name: json['name'] as String,
      algorithmId: (json['algorithm_id'] ?? json['algorithmId']) != null
          ? ((json['algorithm_id'] ?? json['algorithmId']) as num).toInt()
          : 0,
      params: json['params'] as String,
      isSystem: (json['is_system'] ?? json['isSystem']) as bool?,
    );
  }

  Map<String, dynamic> toJson() => {
        if (id != null) 'id': id,
        'name': name,
        'algorithm_id': algorithmId,
        'params': params,
        if (isSystem != null) 'is_system': isSystem,
      };
}

/// 参数预设视图对象
class PresetVO {
  final int id;
  final int? userId;
  final String name;
  final int algorithmId;
  final String params;
  final bool? isSystem;
  final String createTime;

  const PresetVO({
    required this.id,
    this.userId,
    required this.name,
    required this.algorithmId,
    required this.params,
    this.isSystem,
    required this.createTime,
  });

  factory PresetVO.fromJson(Map<String, dynamic> json) {
    return PresetVO(
      id: (json['id'] as num).toInt(),
      userId: (json['user_id'] ?? json['userId']) as int?,
      name: json['name'] as String,
      algorithmId: (json['algorithm_id'] ?? json['algorithmId']) != null
          ? ((json['algorithm_id'] ?? json['algorithmId']) as num).toInt()
          : 0,
      params: json['params'] as String,
      isSystem: (json['is_system'] ?? json['isSystem']) as bool?,
      createTime: json['create_time'] as String? ?? json['createTime'] as String,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        if (userId != null) 'user_id': userId,
        'name': name,
        'algorithm_id': algorithmId,
        'params': params,
        if (isSystem != null) 'is_system': isSystem,
        'create_time': createTime,
      };
}

/// 参数预设查询
class PresetQuery {
  final int? algorithmId;
  final bool? isSystem;
  final int pageNum;
  final int pageSize;

  const PresetQuery({
    this.algorithmId,
    this.isSystem,
    this.pageNum = 1,
    this.pageSize = 20,
  });

  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (algorithmId != null) 'algorithmId': algorithmId,
        if (isSystem != null) 'isSystem': isSystem,
      };
}

// ===== 对比报告 =====

/// 对比报告生成请求
class CompareReportForm {
  final int logId;
  final String format; // "pdf" | "image"
  final bool? includeMetrics;
  final bool? includeFilters;

  const CompareReportForm({
    required this.logId,
    required this.format,
    this.includeMetrics,
    this.includeFilters,
  });

  Map<String, dynamic> toJson() => {
        'log_id': logId,
        'format': format,
        if (includeMetrics != null) 'include_metrics': includeMetrics,
        if (includeFilters != null) 'include_filters': includeFilters,
      };
}

/// 对比报告结果
class CompareReportResult {
  final int taskId;
  final TaskStatus status;
  final String? downloadUrl;
  final String? errorMessage;

  const CompareReportResult({
    required this.taskId,
    required this.status,
    this.downloadUrl,
    this.errorMessage,
  });

  factory CompareReportResult.fromJson(Map<String, dynamic> json) {
    return CompareReportResult(
      taskId: (json['task_id'] ?? json['taskId']) != null
          ? ((json['task_id'] ?? json['taskId']) as num).toInt()
          : 0,
      status: _statusFromJson(
          (json['status'] as num?)?.toInt()),
      downloadUrl:
          (json['download_url'] ?? json['downloadUrl']) as String?,
      errorMessage:
          (json['error_message'] ?? json['errorMessage']) as String?,
    );
  }

  Map<String, dynamic> toJson() => {
        'task_id': taskId,
        'status': _statusToJson(status),
        if (downloadUrl != null) 'download_url': downloadUrl,
        if (errorMessage != null) 'error_message': errorMessage,
      };
}
