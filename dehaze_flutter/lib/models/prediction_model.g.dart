// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'prediction_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictionRequest _$PredictionRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionRequest', json, ($checkedConvert) {
      final val = PredictionRequest(
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        fileId: $checkedConvert('fileId', (v) => v as String),
        params: $checkedConvert('params', (v) => v as Map<String, dynamic>?),
      );
      return val;
    });

Map<String, dynamic> _$PredictionRequestToJson(PredictionRequest instance) =>
    <String, dynamic>{
      'algorithmId': instance.algorithmId,
      'fileId': instance.fileId,
      if (instance.params case final value?) 'params': value,
    };

PredictionResponse _$PredictionResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionResponse', json, ($checkedConvert) {
      final val = PredictionResponse(
        taskId: $checkedConvert('taskId', (v) => v as String),
        status: $checkedConvert(
          'status',
          (v) =>
              $enumDecodeNullable(_$PredictionStatusEnumMap, v) ??
              PredictionStatus.pending,
        ),
        resultUrl: $checkedConvert('resultUrl', (v) => v as String?),
        duration: $checkedConvert('duration', (v) => (v as num?)?.toInt()),
        message: $checkedConvert('message', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PredictionResponseToJson(PredictionResponse instance) =>
    <String, dynamic>{
      'taskId': instance.taskId,
      'status': _$PredictionStatusEnumMap[instance.status]!,
      if (instance.resultUrl case final value?) 'resultUrl': value,
      if (instance.duration case final value?) 'duration': value,
      if (instance.message case final value?) 'message': value,
    };

const _$PredictionStatusEnumMap = {
  PredictionStatus.pending: 'pending',
  PredictionStatus.processing: 'processing',
  PredictionStatus.success: 'success',
  PredictionStatus.failed: 'failed',
};

PredictionLog _$PredictionLogFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionLog', json, ($checkedConvert) {
      final val = PredictionLog(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        originUrl: $checkedConvert('originUrl', (v) => v as String),
        predUrl: $checkedConvert('predUrl', (v) => v as String),
        status: $checkedConvert('status', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        duration: $checkedConvert('duration', (v) => (v as num?)?.toInt()),
        algorithmId: $checkedConvert(
          'algorithmId',
          (v) => (v as num?)?.toInt(),
        ),
        originMd5: $checkedConvert('originMd5', (v) => v as String?),
        predMd5: $checkedConvert('predMd5', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PredictionLogToJson(PredictionLog instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.algorithmId case final value?) 'algorithmId': value,
      'algorithmName': instance.algorithmName,
      'originUrl': instance.originUrl,
      'predUrl': instance.predUrl,
      'status': instance.status,
      if (instance.duration case final value?) 'duration': value,
      'createTime': instance.createTime,
      if (instance.originMd5 case final value?) 'originMd5': value,
      if (instance.predMd5 case final value?) 'predMd5': value,
    };
