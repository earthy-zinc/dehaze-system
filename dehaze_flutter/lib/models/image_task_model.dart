import 'dart:convert';

import 'package:json_annotation/json_annotation.dart';

part 'image_task_model.g.dart';

/// 图像任务状态（后端以整数序列化）
///
/// 1=处理中 / 2=已完成 / 3=失败 / 4=已取消
enum TaskStatus {
  processing,
  completed,
  failed,
  canceled;

  static TaskStatus fromValue(int? value) {
    switch (value) {
      case 2:
        return TaskStatus.completed;
      case 3:
        return TaskStatus.failed;
      case 4:
        return TaskStatus.canceled;
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
    TaskStatus.canceled => 4,
    TaskStatus.processing => 1,
  };
}

/// 图像编辑提交请求
///
/// 对齐后端 ImageEditRequest：model / fileId / imageUrl / params / recommendedBy。
/// fileId 与 imageUrl 至少提供一个；model 为空时取平台默认模型。
@JsonSerializable()
class ImageEditRequest {
  const ImageEditRequest({
    required this.model,
    this.fileId,
    this.imageUrl,
    this.params,
    this.recommendedBy,
  });

  factory ImageEditRequest.fromJson(Map<String, dynamic> json) =>
      _$ImageEditRequestFromJson(json);

  /// 图像模型标识（统一模型规范 slug）
  final String model;

  /// 原始图片文件 ID
  final int? fileId;

  /// 原始图片 URL（与 fileId 二选一）
  final String? imageUrl;

  /// 推理参数
  ///
  /// 后端 `params` 字段为 String 类型，需序列化为 JSON 字符串传输。
  @JsonKey(toJson: _paramsToJson, fromJson: _paramsFromJson)
  final Map<String, dynamic>? params;

  /// 推荐来源：推荐记录 ID（用于追踪推荐采纳率）
  final int? recommendedBy;

  static String? _paramsToJson(Map<String, dynamic>? params) =>
      params == null ? null : jsonEncode(params);

  static Map<String, dynamic>? _paramsFromJson(String? params) =>
      params == null || params.isEmpty
      ? null
      : jsonDecode(params) as Map<String, dynamic>;

  Map<String, dynamic> toJson() => _$ImageEditRequestToJson(this);
}

/// 图像任务响应
///
/// POST 提交返回 taskId + status（缓存命中时 status=2 并携带完整结果）；
/// GET 轮询按 status 返回不同字段。
@JsonSerializable()
class ImageTaskResponse {
  const ImageTaskResponse({
    this.taskId,
    required this.status,
    this.estimatedCredits,
    this.resultUrl,
    this.resultMd5,
    this.originMd5,
    this.resultThumbnailUrl,
    this.time,
    this.errorMessage,
  });

  factory ImageTaskResponse.fromJson(Map<String, dynamic> json) =>
      _$ImageTaskResponseFromJson(json);

  /// 任务 ID（sys_pred_log.id）
  final int? taskId;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus status;

  /// 预估消耗积分（提交时返回）
  final int? estimatedCredits;

  /// 结果图片 URL（completed 时返回）
  final String? resultUrl;

  /// 处理结果图片 MD5（completed 时返回）
  final String? resultMd5;

  /// 原图内容 MD5（completed 时返回）
  final String? originMd5;

  /// 结果缩略图 URL（completed 时返回）
  final String? resultThumbnailUrl;

  /// 处理耗时（秒）
  final int? time;

  /// 失败错误信息（failed 时返回）
  final String? errorMessage;

  Map<String, dynamic> toJson() => _$ImageTaskResponseToJson(this);

  /// 是否已拿到处理结果
  bool get hasResult =>
      status == TaskStatus.completed &&
      resultUrl != null &&
      resultUrl!.isNotEmpty;
}

/// 图像任务历史项（对齐后端 ImageTaskVO）
@JsonSerializable()
class ImageTaskVO {
  const ImageTaskVO({
    required this.id,
    required this.model,
    this.originMd5,
    this.originUrl,
    this.predMd5,
    this.predUrl,
    this.status,
    this.errorMessage,
    this.time,
    this.createTime,
  });

  factory ImageTaskVO.fromJson(Map<String, dynamic> json) =>
      _$ImageTaskVOFromJson(json);

  final int id;

  /// 图像模型标识（统一模型规范 slug）
  final String model;

  /// 原图内容 MD5
  final String? originMd5;

  /// 原图 URL
  final String? originUrl;

  /// 结果图内容 MD5
  final String? predMd5;

  /// 结果图 URL
  final String? predUrl;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus? status;

  /// 失败错误信息
  final String? errorMessage;

  /// 处理耗时（秒）
  final int? time;

  /// 创建时间
  final String? createTime;

  Map<String, dynamic> toJson() => _$ImageTaskVOToJson(this);
}

// ===== 参数预设 =====

/// 参数预设表单
@JsonSerializable()
class PresetForm {
  const PresetForm({
    this.id,
    required this.name,
    required this.modelId,
    this.params,
    this.isDefault,
  });

  factory PresetForm.fromJson(Map<String, dynamic> json) =>
      _$PresetFormFromJson(json);

  final int? id;
  final String name;

  /// 关联图像模型标识（统一模型规范 slug）
  final String modelId;

  /// 参数键值对（JSON 字符串）
  final String? params;

  /// 是否默认预设
  final int? isDefault;

  Map<String, dynamic> toJson() => _$PresetFormToJson(this);
}

/// 参数预设视图对象
@JsonSerializable()
class PresetVO {
  const PresetVO({
    required this.id,
    required this.name,
    required this.type,
    required this.modelId,
    this.params,
    this.userId,
    required this.isDefault,
    this.createTime,
  });

  factory PresetVO.fromJson(Map<String, dynamic> json) =>
      _$PresetVOFromJson(json);

  final int id;
  final String name;

  /// 预设类型（system:系统预设; custom:用户自定义）
  final String type;

  /// 关联图像模型标识（统一模型规范 slug）
  final String modelId;

  final String? params;
  final int? userId;
  final int isDefault;
  final String? createTime;

  Map<String, dynamic> toJson() => _$PresetVOToJson(this);
}

/// 参数预设查询
@JsonSerializable()
class PresetQuery {
  const PresetQuery({
    this.modelId,
    this.type,
    this.pageNum = 1,
    this.pageSize = 20,
  });

  factory PresetQuery.fromJson(Map<String, dynamic> json) =>
      _$PresetQueryFromJson(json);

  /// 按图像模型标识筛选
  final String? modelId;

  /// 预设类型筛选
  final String? type;

  final int pageNum;
  final int pageSize;

  Map<String, dynamic> toJson() => _$PresetQueryToJson(this);

  Map<String, dynamic> toQuery() => {
    'pageNum': pageNum,
    'pageSize': pageSize,
    if (modelId != null) 'modelId': modelId,
    if (type != null) 'type': type,
  };
}

// ===== 对比报告 =====
//
// 对比报告仍在 /compare/report 端点（未随图像任务改造迁移）。

/// 对比报告生成请求
@JsonSerializable()
class CompareReportForm {
  const CompareReportForm({
    required this.logId,
    required this.format,
    this.includeMetrics,
    this.includeFilters,
  });

  factory CompareReportForm.fromJson(Map<String, dynamic> json) =>
      _$CompareReportFormFromJson(json);

  final int logId;
  final String format;
  final bool? includeMetrics;
  final bool? includeFilters;

  Map<String, dynamic> toJson() => _$CompareReportFormToJson(this);
}

/// 对比报告结果
@JsonSerializable()
class CompareReportResult {
  const CompareReportResult({
    required this.taskId,
    required this.status,
    this.downloadUrl,
    this.errorMessage,
  });

  factory CompareReportResult.fromJson(Map<String, dynamic> json) =>
      _$CompareReportResultFromJson(json);

  final int taskId;

  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus status;

  final String? downloadUrl;
  final String? errorMessage;

  Map<String, dynamic> toJson() => _$CompareReportResultToJson(this);
}
