// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'evaluation_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

EvaluationRequest _$EvaluationRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationRequest', json, ($checkedConvert) {
      final val = EvaluationRequest(
        modelId: $checkedConvert('modelId', (v) => v as String),
        predUrl: $checkedConvert('predUrl', (v) => v as String?),
        gtUrl: $checkedConvert('gtUrl', (v) => v as String?),
        params: $checkedConvert('params', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationRequestToJson(EvaluationRequest instance) =>
    <String, dynamic>{
      'modelId': instance.modelId,
      'predUrl': ?instance.predUrl,
      'gtUrl': ?instance.gtUrl,
      'params': ?instance.params,
    };

EvaluationResult _$EvaluationResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationResult', json, ($checkedConvert) {
      final val = EvaluationResult(
        logId: $checkedConvert('logId', (v) => (v as num?)?.toInt()),
        status: $checkedConvert(
          'status',
          (v) => _evalStatusFromJson((v as num?)?.toInt()),
        ),
        metrics: $checkedConvert(
          'metrics',
          (v) =>
              (v as Map<String, dynamic>?)?.map(
                (k, e) => MapEntry(k, (e as num).toDouble()),
              ) ??
              {},
        ),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationResultToJson(EvaluationResult instance) =>
    <String, dynamic>{
      'logId': ?instance.logId,
      'status': ?_evalStatusToJson(instance.status),
      'metrics': ?instance.metrics,
      'time': ?instance.time,
      'errorMessage': ?instance.errorMessage,
    };

EvaluationLog _$EvaluationLogFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationLog', json, ($checkedConvert) {
      final val = EvaluationLog(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        modelId: $checkedConvert('modelId', (v) => v as String),
        predUrl: $checkedConvert('predUrl', (v) => v as String?),
        gtUrl: $checkedConvert('gtUrl', (v) => v as String?),
        status: $checkedConvert(
          'status',
          (v) => _evalStatusFromJson((v as num?)?.toInt()),
        ),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        result: $checkedConvert('result', (v) => v),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$EvaluationLogToJson(EvaluationLog instance) =>
    <String, dynamic>{
      'id': instance.id,
      'modelId': instance.modelId,
      'predUrl': ?instance.predUrl,
      'gtUrl': ?instance.gtUrl,
      'status': ?_evalStatusToJson(instance.status),
      'errorMessage': ?instance.errorMessage,
      'result': ?instance.result,
      'time': ?instance.time,
      'createTime': ?instance.createTime,
    };
