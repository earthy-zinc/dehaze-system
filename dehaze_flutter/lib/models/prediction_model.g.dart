// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'prediction_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PredictionRequest _$PredictionRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionRequest', json, ($checkedConvert) {
      final val = PredictionRequest(
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        fileId: $checkedConvert('fileId', (v) => (v as num?)?.toInt()),
        imageUrl: $checkedConvert('imageUrl', (v) => v as String?),
        params: $checkedConvert(
          'params',
          (v) => PredictionRequest._paramsFromJson(v as String?),
        ),
        recommendedBy: $checkedConvert(
          'recommendedBy',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$PredictionRequestToJson(PredictionRequest instance) =>
    <String, dynamic>{
      'algorithmId': instance.algorithmId,
      if (instance.fileId case final value?) 'fileId': value,
      if (instance.imageUrl case final value?) 'imageUrl': value,
      if (PredictionRequest._paramsToJson(instance.params) case final value?)
        'params': value,
      if (instance.recommendedBy case final value?) 'recommendedBy': value,
    };

PredictionResponse _$PredictionResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionResponse', json, ($checkedConvert) {
      final val = PredictionResponse(
        logId: $checkedConvert('logId', (v) => (v as num?)?.toInt()),
        status: $checkedConvert(
          'status',
          (v) => _statusFromJson((v as num?)?.toInt()),
        ),
        resultUrl: $checkedConvert('resultUrl', (v) => v as String?),
        resultThumbnailUrl: $checkedConvert(
          'resultThumbnailUrl',
          (v) => v as String?,
        ),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        fromCache: $checkedConvert('fromCache', (v) => v as bool?),
      );
      return val;
    });

Map<String, dynamic> _$PredictionResponseToJson(PredictionResponse instance) =>
    <String, dynamic>{
      if (instance.logId case final value?) 'logId': value,
      if (_statusToJson(instance.status) case final value?) 'status': value,
      if (instance.resultUrl case final value?) 'resultUrl': value,
      if (instance.resultThumbnailUrl case final value?)
        'resultThumbnailUrl': value,
      if (instance.time case final value?) 'time': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
      if (instance.fromCache case final value?) 'fromCache': value,
    };

PredictionLog _$PredictionLogFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PredictionLog', json, ($checkedConvert) {
      final val = PredictionLog(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        algorithmId: $checkedConvert(
          'algorithmId',
          (v) => (v as num?)?.toInt(),
        ),
        originUrl: $checkedConvert('originUrl', (v) => v as String?),
        predUrl: $checkedConvert('predUrl', (v) => v as String?),
        status: $checkedConvert(
          'status',
          (v) => _statusFromJson((v as num?)?.toInt()),
        ),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$PredictionLogToJson(PredictionLog instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.algorithmId case final value?) 'algorithmId': value,
      'algorithmName': instance.algorithmName,
      if (instance.originUrl case final value?) 'originUrl': value,
      if (instance.predUrl case final value?) 'predUrl': value,
      if (_statusToJson(instance.status) case final value?) 'status': value,
      if (instance.errorMessage case final value?) 'errorMessage': value,
      if (instance.time case final value?) 'time': value,
      'createTime': instance.createTime,
    };
