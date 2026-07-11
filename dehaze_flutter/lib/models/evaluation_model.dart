import 'package:json_annotation/json_annotation.dart';

part 'evaluation_model.g.dart';

/// 评估请求
@JsonSerializable()
class EvaluationRequest {
  const EvaluationRequest({
    required this.algorithmId,
    required this.predFileId,
    required this.gtFileId,
  });

  factory EvaluationRequest.fromJson(Map<String, dynamic> json) =>
      _$EvaluationRequestFromJson(json);

  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  /// 预测结果图文件 ID
  @JsonKey(name: 'predFileId')
  final String predFileId;

  /// 真值图（Ground Truth）文件 ID
  @JsonKey(name: 'gtFileId')
  final String gtFileId;

  Map<String, dynamic> toJson() => _$EvaluationRequestToJson(this);
}

/// 评估结果
@JsonSerializable()
class EvaluationResult {
  const EvaluationResult({
    required this.taskId,
    this.status = 'pending',
    this.metrics,
    this.message,
  });

  factory EvaluationResult.fromJson(Map<String, dynamic> json) =>
      _$EvaluationResultFromJson(json);

  @JsonKey(name: 'taskId')
  final String taskId;

  /// 任务状态（pending/processing/success/failed）
  final String status;

  /// 评估指标
  final EvaluationMetrics? metrics;

  final String? message;

  Map<String, dynamic> toJson() => _$EvaluationResultToJson(this);

  bool get isCompleted => status == 'success' || status == 'failed';
}

/// 评估指标
@JsonSerializable()
class EvaluationMetrics {
  const EvaluationMetrics({
    this.psnr,
    this.ssim,
    this.mse,
    this.fsim,
    this.lpips,
  });

  factory EvaluationMetrics.fromJson(Map<String, dynamic> json) =>
      _$EvaluationMetricsFromJson(json);

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

  Map<String, dynamic> toJson() => _$EvaluationMetricsToJson(this);

  /// 获取所有指标列表（用于 UI 展示）
  List<MetricItem> toList() => [
        MetricItem(name: 'PSNR', value: psnr, unit: 'dB', higherIsBetter: true, description: '峰值信噪比'),
        MetricItem(name: 'SSIM', value: ssim, unit: '', higherIsBetter: true, description: '结构相似性'),
        MetricItem(name: 'MSE', value: mse, unit: '', higherIsBetter: false, description: '均方误差'),
        MetricItem(name: 'FSIM', value: fsim, unit: '', higherIsBetter: true, description: '特征相似性'),
        MetricItem(name: 'LPIPS', value: lpips, unit: '', higherIsBetter: false, description: '感知损失'),
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

  String get displayValue => value != null
      ? value!.toStringAsFixed(value! < 1 ? 4 : 2)
      : '-';
}
