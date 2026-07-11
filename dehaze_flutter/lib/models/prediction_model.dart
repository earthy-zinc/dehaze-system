import 'package:json_annotation/json_annotation.dart';

part 'prediction_model.g.dart';

/// 预测任务状态枚举
enum PredictionStatus {
  @JsonValue('pending')
  pending,
  @JsonValue('processing')
  processing,
  @JsonValue('success')
  success,
  @JsonValue('failed')
  failed,
}

extension PredictionStatusExtension on PredictionStatus {
  String get displayName {
    switch (this) {
      case PredictionStatus.pending:
        return '等待中';
      case PredictionStatus.processing:
        return '处理中';
      case PredictionStatus.success:
        return '已完成';
      case PredictionStatus.failed:
        return '处理失败';
    }
  }

  bool get isCompleted =>
      this == PredictionStatus.success || this == PredictionStatus.failed;
}

/// 预测请求
@JsonSerializable()
class PredictionRequest {
  const PredictionRequest({
    required this.algorithmId,
    required this.fileId,
    this.params,
  });

  factory PredictionRequest.fromJson(Map<String, dynamic> json) =>
      _$PredictionRequestFromJson(json);

  /// 算法 ID
  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  /// 原始图片文件 ID
  @JsonKey(name: 'fileId')
  final String fileId;

  /// 算法参数（自定义 JSON）
  final Map<String, dynamic>? params;

  Map<String, dynamic> toJson() => _$PredictionRequestToJson(this);
}

/// 预测响应
@JsonSerializable()
class PredictionResponse {
  const PredictionResponse({
    required this.taskId,
    this.status = PredictionStatus.pending,
    this.resultUrl,
    this.duration,
    this.message,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) =>
      _$PredictionResponseFromJson(json);

  /// 任务 ID
  @JsonKey(name: 'taskId')
  final String taskId;

  /// 任务状态
  @JsonKey(defaultValue: PredictionStatus.pending)
  final PredictionStatus status;

  /// 结果图片 URL
  @JsonKey(name: 'resultUrl')
  final String? resultUrl;

  /// 处理耗时（毫秒）
  final int? duration;

  /// 消息
  final String? message;

  Map<String, dynamic> toJson() => _$PredictionResponseToJson(this);

  /// 是否已完成
  bool get isCompleted => status.isCompleted;
}

/// 预测日志
@JsonSerializable()
class PredictionLog {
  const PredictionLog({
    required this.id,
    required this.algorithmName,
    required this.originUrl,
    required this.predUrl,
    required this.status,
    required this.createTime,
    this.duration,
    this.algorithmId,
    this.originMd5,
    this.predMd5,
  });

  factory PredictionLog.fromJson(Map<String, dynamic> json) =>
      _$PredictionLogFromJson(json);

  final int id;

  @JsonKey(name: 'algorithmId')
  final int? algorithmId;

  @JsonKey(name: 'algorithmName')
  final String algorithmName;

  /// 原始图片 URL
  @JsonKey(name: 'originUrl')
  final String originUrl;

  /// 预测结果图片 URL
  @JsonKey(name: 'predUrl')
  final String predUrl;

  /// 状态
  final String status;

  /// 耗时（毫秒）
  final int? duration;

  /// 创建时间
  @JsonKey(name: 'createTime')
  final String createTime;

  @JsonKey(name: 'originMd5')
  final String? originMd5;

  @JsonKey(name: 'predMd5')
  final String? predMd5;

  Map<String, dynamic> toJson() => _$PredictionLogToJson(this);
}
