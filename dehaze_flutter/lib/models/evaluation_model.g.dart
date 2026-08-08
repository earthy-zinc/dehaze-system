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
      if (instance.logId case final value?) 'logId': value,
      if (_evalStatusToJson(instance.status) case final value?) 'status': value,
      if (instance.metrics case final value?) 'metrics': value,
      if (instance.time case final value?) 'time': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
    };

EvaluationLog _$EvaluationLogFromJson(Map<String, dynamic> json) =>
    $checkedCreate('EvaluationLog', json, ($checkedConvert) {
      final val = EvaluationLog(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String?),
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
      'algorithmId': instance.algorithmId,
      if (instance.algorithmName case final value?) 'algorithmName': value,
      if (instance.predUrl case final value?) 'predUrl': value,
      if (instance.gtUrl case final value?) 'gtUrl': value,
      if (_evalStatusToJson(instance.status) case final value?) 'status': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
      if (instance.result case final value?) 'result': value,
      if (instance.time case final value?) 'time': value,
      if (instance.createTime case final value?) 'createTime': value,
    };
