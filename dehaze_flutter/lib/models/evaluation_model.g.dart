// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'evaluation_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

EvaluationRequest _$EvaluationRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationRequest', json, ($checkedConvert) {
      final val = EvaluationRequest(
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        predFileId: $checkedConvert('predFileId', (v) => v as String),
        gtFileId: $checkedConvert('gtFileId', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationRequestToJson(EvaluationRequest instance) =>
    <String, dynamic>{
      'algorithmId': instance.algorithmId,
      'predFileId': instance.predFileId,
      'gtFileId': instance.gtFileId,
    };

EvaluationResult _$EvaluationResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationResult', json, ($checkedConvert) {
      final val = EvaluationResult(
        taskId: $checkedConvert('taskId', (v) => v as String),
        status: $checkedConvert('status', (v) => v as String? ?? 'pending'),
        metrics: $checkedConvert(
          'metrics',
          (v) => v == null
              ? null
              : EvaluationMetrics.fromJson(v as Map<String, dynamic>),
        ),
        message: $checkedConvert('message', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationResultToJson(EvaluationResult instance) =>
    <String, dynamic>{
      'taskId': instance.taskId,
      'status': instance.status,
      if (instance.metrics?.toJson() case final value?) 'metrics': value,
      if (instance.message case final value?) 'message': value,
    };

EvaluationMetrics _$EvaluationMetricsFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationMetrics', json, ($checkedConvert) {
      final val = EvaluationMetrics(
        psnr: $checkedConvert('psnr', (v) => (v as num?)?.toDouble()),
        ssim: $checkedConvert('ssim', (v) => (v as num?)?.toDouble()),
        mse: $checkedConvert('mse', (v) => (v as num?)?.toDouble()),
        fsim: $checkedConvert('fsim', (v) => (v as num?)?.toDouble()),
        lpips: $checkedConvert('lpips', (v) => (v as num?)?.toDouble()),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationMetricsToJson(EvaluationMetrics instance) =>
    <String, dynamic>{
      if (instance.psnr case final value?) 'psnr': value,
      if (instance.ssim case final value?) 'ssim': value,
      if (instance.mse case final value?) 'mse': value,
      if (instance.fsim case final value?) 'fsim': value,
      if (instance.lpips case final value?) 'lpips': value,
    };
