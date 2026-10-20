import 'package:json_annotation/json_annotation.dart';

import 'image_task_model.dart';

part 'evaluation_model.g.dart';

TaskStatus _evalStatusFromJson(int? value) => TaskStatus.fromValue(value);

int? _evalStatusToJson(TaskStatus? status) {
  if (status == null) return null;
  return switch (status) {
    TaskStatus.completed => 2,
    TaskStatus.failed => 3,
    TaskStatus.canceled => 4,
    TaskStatus.processing => 1,
  };
}

/// 评估请求
///
/// 对应后端 EvaluationForm：
/// modelId（图像模型 slug，必填）、predUrl/gtUrl（String）、params（String，JSON）。
/// 去雾流程中使用结果图 URL 进行评估。
@JsonSerializable(includeIfNull: false)
class EvaluationRequest {
  const EvaluationRequest({
    required this.modelId,
    this.predUrl,
    this.gtUrl,
    this.params,
  });

  factory EvaluationRequest.fromJson(Map<String, dynamic> json) =>
      _$EvaluationRequestFromJson(json);

  /// 图像模型标识（统一模型规范 slug）
  final String modelId;

  /// 预测结果图 URL
  final String? predUrl;

  /// 真值图（Ground Truth）URL
  final String? gtUrl;

  /// 评估参数（JSON 字符串）
  final String? params;

  Map<String, dynamic> toJson() => _$EvaluationRequestToJson(this);
}

/// 评估结果
///
/// 对应后端 EvaluationResultVO：异步任务模式。
/// POST 立即返回 logId + status=processing；GET 轮询至 completed/failed 时返回完整字段。
@JsonSerializable()
class EvaluationResult {
  const EvaluationResult({
    this.logId,
    required this.status,
    this.metrics,
    this.time,
    this.errorMessage,
  });

  factory EvaluationResult.fromJson(Map<String, dynamic> json) =>
      _$EvaluationResultFromJson(json);

  /// 评估日志 ID（POST 返回时可能为 null）
  final int? logId;

  /// 任务状态
  @JsonKey(fromJson: _evalStatusFromJson, toJson: _evalStatusToJson)
  final TaskStatus status;

  /// 指标结果（PSNR/SSIM/MSE/FSIM/LPIPS 等，status=completed 时返回）
  @JsonKey(defaultValue: <String, double>{})
  final Map<String, double>? metrics;

  /// 处理耗时（秒）
  final int? time;

  /// 失败错误信息（status=failed 时返回）
  final String? errorMessage;

  Map<String, dynamic> toJson() => _$EvaluationResultToJson(this);

  /// 转换为结构化指标模型（供 UI 展示）
  EvaluationMetrics get metricsModel =>
      EvaluationMetrics.fromMap(metrics ?? const {});
}

/// 评估指标（结构化展示模型）
class EvaluationMetrics {
  const EvaluationMetrics({
    this.psnr,
    this.ssim,
    this.mse,
    this.fsim,
    this.lpips,
  });

  /// 从后端指标 Map 构建（键名不区分大小写，容忍缺失）
  factory EvaluationMetrics.fromMap(Map<String, double> map) {
    double? pick(String key) {
      for (final entry in map.entries) {
        if (entry.key.toLowerCase() == key) return entry.value;
      }
      return null;
    }

    return EvaluationMetrics(
      psnr: pick('psnr'),
      ssim: pick('ssim'),
      mse: pick('mse'),
      fsim: pick('fsim'),
      lpips: pick('lpips'),
    );
  }

  /// 峰值信噪比（越高越好，通常 20-40 dB）
  final double? psnr;

  /// 结构相似性（0-1，越高越好）
  final double? ssim;

  /// 均方误差（越低越好）
  final double? mse;

  /// 特征相似性（0-1，越高越好）
  final double? fsim;

  /// 感知损失（越低越好）
  final double? lpips;

  /// 获取所有指标列表（用于 UI 展示）
  List<MetricItem> toList() => [
    MetricItem(
      name: 'PSNR',
      value: psnr,
      unit: 'dB',
      higherIsBetter: true,
      description: '峰值信噪比',
    ),
    MetricItem(
      name: 'SSIM',
      value: ssim,
      unit: '',
      higherIsBetter: true,
      description: '结构相似性',
    ),
    MetricItem(
      name: 'MSE',
      value: mse,
      unit: '',
      higherIsBetter: false,
      description: '均方误差',
    ),
    MetricItem(
      name: 'FSIM',
      value: fsim,
      unit: '',
      higherIsBetter: true,
      description: '特征相似性',
    ),
    MetricItem(
      name: 'LPIPS',
      value: lpips,
      unit: '',
      higherIsBetter: false,
      description: '感知损失',
    ),
  ];
}

/// 单个指标项（UI 展示用）
class MetricItem {
  const MetricItem({
    required this.name,
    required this.value,
    required this.unit,
    required this.higherIsBetter,
    required this.description,
  });

  final String name;
  final double? value;
  final String unit;
  final bool higherIsBetter;
  final String description;

  String get displayValue =>
      value != null ? value!.toStringAsFixed(value! < 1 ? 4 : 2) : '-';
}

/// 评估日志（对应后端 EvalLogVO）
///
/// 用于评估历史列表展示。
@JsonSerializable()
class EvaluationLog {
  const EvaluationLog({
    required this.id,
    required this.modelId,
    this.predUrl,
    this.gtUrl,
    this.status,
    this.errorMessage,
    this.result,
    this.time,
    this.createTime,
  });

  factory EvaluationLog.fromJson(Map<String, dynamic> json) =>
      _$EvaluationLogFromJson(json);

  final int id;

  /// 图像模型标识（统一模型规范 slug）
  final String modelId;

  /// 预测结果图 URL
  final String? predUrl;

  /// 真值图 URL
  final String? gtUrl;

  /// 任务状态
  @JsonKey(fromJson: _evalStatusFromJson, toJson: _evalStatusToJson)
  final TaskStatus? status;

  /// 失败错误信息
  final String? errorMessage;

  /// 评估指标结果（PSNR/SSIM 等 Map 或 JSON 字符串）
  final dynamic result;

  /// 处理耗时（秒）
  final int? time;

  /// 创建时间
  final String? createTime;

  Map<String, dynamic> toJson() => _$EvaluationLogToJson(this);
}

/// 评估指标历史 VO（对应 JS SDK EvalMetricsVO）
///
/// 用于评估指标历史列表展示，仅包含已完成任务的指标数据。
class EvalMetricsVO {
  final int id;
  final String modelId;
  final String? predUrl;
  final String? gtUrl;
  final Map<String, double>? metrics;
  final int? time;
  final TaskStatus? status;
  final String? errorMessage;
  final String? createTime;

  const EvalMetricsVO({
    required this.id,
    required this.modelId,
    this.predUrl,
    this.gtUrl,
    this.metrics,
    this.time,
    this.status,
    this.errorMessage,
    this.createTime,
  });

  factory EvalMetricsVO.fromJson(Map<String, dynamic> json) {
    return EvalMetricsVO(
      id: (json['id'] as num).toInt(),
      modelId: (json['modelId'] ?? json['model_id']) as String? ?? '',
      predUrl: (json['predUrl'] ?? json['pred_url']) as String?,
      gtUrl: (json['gtUrl'] ?? json['gt_url']) as String?,
      metrics: (json['metrics'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toDouble()),
      ),
      time: (json['time'] as num?)?.toInt(),
      status: _evalStatusFromJson((json['status'] as num?)?.toInt()),
      errorMessage: (json['errorMessage'] ?? json['error_message']) as String?,
      createTime: (json['createTime'] ?? json['create_time']) as String?,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'modelId': modelId,
    if (predUrl != null) 'predUrl': predUrl,
    if (gtUrl != null) 'gtUrl': gtUrl,
    if (metrics != null) 'metrics': metrics,
    if (time != null) 'time': time,
    if (status != null) 'status': _evalStatusToJson(status),
    if (errorMessage != null) 'errorMessage': errorMessage,
    if (createTime != null) 'createTime': createTime,
  };
}
