// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'evaluation_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

EvaluationRequest _$EvaluationRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationRequest', json, ($checkedConvert) {
      final val = EvaluationRequest(
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        predUrl: $checkedConvert('predUrl', (v) => v as String?),
        gtUrl: $checkedConvert('gtUrl', (v) => v as String?),
        params: $checkedConvert('params', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationRequestToJson(EvaluationRequest instance) =>
    <String, dynamic>{
      'algorithmId': instance.algorithmId,
      if (instance.predUrl case final value?) 'predUrl': value,
      if (instance.gtUrl case final value?) 'gtUrl': value,
      if (instance.params case final value?) 'params': value,
    };

EvaluationResult _$EvaluationResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationResult', json, ($checkedConvert) {
      final val = EvaluationResult(
        logId: $checkedConvert('logId', (v) => (v as num).toInt()),
        metrics: $checkedConvert(
          'metrics',
          (v) =>
              (v as Map<String, dynamic>?)?.map(
                (k, e) => MapEntry(k, (e as num).toDouble()),
              ) ??
              {},
        ),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationResultToJson(EvaluationResult instance) =>
    <String, dynamic>{
      'logId': instance.logId,
      'metrics': instance.metrics,
      if (instance.time case final value?) 'time': value,
    };
