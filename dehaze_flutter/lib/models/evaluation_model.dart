import 'package:json_annotation/json_annotation.dart';

import 'prediction_model.dart';

part 'evaluation_model.g.dart';

/// 评估请求
///
/// 对应后端 EvaluationForm：
/// algorithmId（Long，必填）、predUrl/gtUrl（String）、params（String，JSON）。
/// 去雾流程中使用结果图 URL 进行评估。
@JsonSerializable(includeIfNull: false)
class EvaluationRequest {
  const EvaluationRequest({
    required this.algorithmId,
    this.predUrl,
    this.gtUrl,
    this.params,
  });

  factory EvaluationRequest.fromJson(Map<String, dynamic> json) =>
      _$EvaluationRequestFromJson(json);

  /// 算法 ID
  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  /// 预测结果图 URL
  @JsonKey(name: 'predUrl')
  final String? predUrl;

  /// 真值图（Ground Truth）URL
  @JsonKey(name: 'gtUrl')
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
    required this.logId,
    required this.status,
    this.metrics,
    this.time,
    this.errorMessage,
  });

  factory EvaluationResult.fromJson(Map<String, dynamic> json) =>
      _$EvaluationResultFromJson(json);

  /// 评估日志 ID
  @JsonKey(name: 'logId')
  final int logId;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatus status;

  /// 指标结果（PSNR/SSIM/MSE/FSIM/LPIPS 等，status=completed 时返回）
  @JsonKey(defaultValue: <String, double>{})
  final Map<String, double>? metrics;

  /// 处理耗时（毫秒）
  final int? time;

  /// 失败错误信息（status=failed 时返回）
  @JsonKey(name: 'errorMessage')
  final String? errorMessage;

  static TaskStatus _statusFromJson(int? value) =>
      TaskStatus.fromValue(value);

  static int _statusToJson(TaskStatus status) => switch (status) {
        TaskStatus.completed => 2,
        TaskStatus.failed => 3,
        TaskStatus.processing => 1,
      };

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
