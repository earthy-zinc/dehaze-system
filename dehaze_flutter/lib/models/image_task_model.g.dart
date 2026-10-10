// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'image_task_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ImageEditRequest _$ImageEditRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImageEditRequest', json, ($checkedConvert) {
      final val = ImageEditRequest(
        model: $checkedConvert('model', (v) => v as String),
        fileId: $checkedConvert('fileId', (v) => (v as num?)?.toInt()),
        imageUrl: $checkedConvert('imageUrl', (v) => v as String?),
        params: $checkedConvert(
          'params',
          (v) => ImageEditRequest._paramsFromJson(v as String?),
        ),
        recommendedBy: $checkedConvert(
          'recommendedBy',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ImageEditRequestToJson(ImageEditRequest instance) =>
    <String, dynamic>{
      'model': instance.model,
      'fileId': ?instance.fileId,
      'imageUrl': ?instance.imageUrl,
      'params': ?ImageEditRequest._paramsToJson(instance.params),
      'recommendedBy': ?instance.recommendedBy,
    };

ImageTaskResponse _$ImageTaskResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImageTaskResponse', json, ($checkedConvert) {
      final val = ImageTaskResponse(
        taskId: $checkedConvert('taskId', (v) => (v as num?)?.toInt()),
        status: $checkedConvert(
          'status',
          (v) => _statusFromJson((v as num?)?.toInt()),
        ),
        estimatedCredits: $checkedConvert(
          'estimatedCredits',
          (v) => (v as num?)?.toInt(),
        ),
        resultUrl: $checkedConvert('resultUrl', (v) => v as String?),
        resultMd5: $checkedConvert('resultMd5', (v) => v as String?),
        originMd5: $checkedConvert('originMd5', (v) => v as String?),
        resultThumbnailUrl: $checkedConvert(
          'resultThumbnailUrl',
          (v) => v as String?,
        ),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ImageTaskResponseToJson(ImageTaskResponse instance) =>
    <String, dynamic>{
      'taskId': ?instance.taskId,
      'status': ?_statusToJson(instance.status),
      'estimatedCredits': ?instance.estimatedCredits,
      'resultUrl': ?instance.resultUrl,
      'resultMd5': ?instance.resultMd5,
      'originMd5': ?instance.originMd5,
      'resultThumbnailUrl': ?instance.resultThumbnailUrl,
      'time': ?instance.time,
      'errorMessage': ?instance.errorMessage,
    };

ImageTaskVO _$ImageTaskVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ImageTaskVO', json, ($checkedConvert) {
      final val = ImageTaskVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        model: $checkedConvert('model', (v) => v as String),
        originMd5: $checkedConvert('originMd5', (v) => v as String?),
        originUrl: $checkedConvert('originUrl', (v) => v as String?),
        predMd5: $checkedConvert('predMd5', (v) => v as String?),
        predUrl: $checkedConvert('predUrl', (v) => v as String?),
        status: $checkedConvert(
          'status',
          (v) => _statusFromJson((v as num?)?.toInt()),
        ),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
        time: $checkedConvert('time', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ImageTaskVOToJson(ImageTaskVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'model': instance.model,
      'originMd5': ?instance.originMd5,
      'originUrl': ?instance.originUrl,
      'predMd5': ?instance.predMd5,
      'predUrl': ?instance.predUrl,
      'status': ?_statusToJson(instance.status),
      'errorMessage': ?instance.errorMessage,
      'time': ?instance.time,
      'createTime': ?instance.createTime,
    };

PresetForm _$PresetFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PresetForm', json, ($checkedConvert) {
      final val = PresetForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        name: $checkedConvert('name', (v) => v as String),
        modelId: $checkedConvert('modelId', (v) => v as String),
        params: $checkedConvert('params', (v) => v as String?),
        isDefault: $checkedConvert('isDefault', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$PresetFormToJson(PresetForm instance) =>
    <String, dynamic>{
      'id': ?instance.id,
      'name': instance.name,
      'modelId': instance.modelId,
      'params': ?instance.params,
      'isDefault': ?instance.isDefault,
    };

PresetVO _$PresetVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PresetVO', json, ($checkedConvert) {
      final val = PresetVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        modelId: $checkedConvert('modelId', (v) => v as String),
        params: $checkedConvert('params', (v) => v as String?),
        userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
        isDefault: $checkedConvert('isDefault', (v) => (v as num).toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PresetVOToJson(PresetVO instance) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'type': instance.type,
  'modelId': instance.modelId,
  'params': ?instance.params,
  'userId': ?instance.userId,
  'isDefault': instance.isDefault,
  'createTime': ?instance.createTime,
};

PresetQuery _$PresetQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'PresetQuery',
  json,
  ($checkedConvert) {
    final val = PresetQuery(
      modelId: $checkedConvert('modelId', (v) => v as String?),
      type: $checkedConvert('type', (v) => v as String?),
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 20),
    );
    return val;
  },
);

Map<String, dynamic> _$PresetQueryToJson(PresetQuery instance) =>
    <String, dynamic>{
      'modelId': ?instance.modelId,
      'type': ?instance.type,
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
    };

CompareReportForm _$CompareReportFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CompareReportForm', json, ($checkedConvert) {
      final val = CompareReportForm(
        logId: $checkedConvert('logId', (v) => (v as num).toInt()),
        format: $checkedConvert('format', (v) => v as String),
        includeMetrics: $checkedConvert('includeMetrics', (v) => v as bool?),
        includeFilters: $checkedConvert('includeFilters', (v) => v as bool?),
      );
      return val;
    });

Map<String, dynamic> _$CompareReportFormToJson(CompareReportForm instance) =>
    <String, dynamic>{
      'logId': instance.logId,
      'format': instance.format,
      'includeMetrics': ?instance.includeMetrics,
      'includeFilters': ?instance.includeFilters,
    };

CompareReportResult _$CompareReportResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CompareReportResult', json, ($checkedConvert) {
      final val = CompareReportResult(
        taskId: $checkedConvert('taskId', (v) => (v as num).toInt()),
        status: $checkedConvert(
          'status',
          (v) => _statusFromJson((v as num?)?.toInt()),
        ),
        downloadUrl: $checkedConvert('downloadUrl', (v) => v as String?),
        errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$CompareReportResultToJson(
  CompareReportResult instance,
) => <String, dynamic>{
  'taskId': instance.taskId,
  'status': ?_statusToJson(instance.status),
  'downloadUrl': ?instance.downloadUrl,
  'errorMessage': ?instance.errorMessage,
};
