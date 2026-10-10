// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'task_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TaskVO _$TaskVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('TaskVO', json, ($checkedConvert) {
      final val = TaskVO(
        taskId: $checkedConvert('taskId', (v) => v as String),
        status: $checkedConvert('status', (v) => _statusFromJson(v)),
        progress: $checkedConvert('progress', (v) => (v as num).toInt()),
        taskType: $checkedConvert('taskType', (v) => v as String?),
        taskCategory: $checkedConvert(
          'taskCategory',
          (v) => _categoryFromJson(v),
        ),
        totalFiles: $checkedConvert('totalFiles', (v) => (v as num?)?.toInt()),
        processedFiles: $checkedConvert(
          'processedFiles',
          (v) => (v as num?)?.toInt(),
        ),
        downloadUrl: $checkedConvert('downloadUrl', (v) => v as String?),
        expiresAt: $checkedConvert('expiresAt', (v) => v as String?),
        createdAt: $checkedConvert('createdAt', (v) => v as String?),
        startedAt: $checkedConvert('startedAt', (v) => v as String?),
        completedAt: $checkedConvert('completedAt', (v) => v as String?),
        error: $checkedConvert('error', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$TaskVOToJson(TaskVO instance) => <String, dynamic>{
  'taskId': instance.taskId,
  'status': ?_statusToJson(instance.status),
  'progress': instance.progress,
  'taskType': ?instance.taskType,
  'taskCategory': ?_categoryToJson(instance.taskCategory),
  'totalFiles': ?instance.totalFiles,
  'processedFiles': ?instance.processedFiles,
  'downloadUrl': ?instance.downloadUrl,
  'expiresAt': ?instance.expiresAt,
  'createdAt': ?instance.createdAt,
  'startedAt': ?instance.startedAt,
  'completedAt': ?instance.completedAt,
  'error': ?instance.error,
};

TaskCreateForm _$TaskCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('TaskCreateForm', json, ($checkedConvert) {
      final val = TaskCreateForm(
        type: $checkedConvert('type', (v) => v as String),
        targetId: $checkedConvert('targetId', (v) => (v as num?)?.toInt()),
        targetIds: $checkedConvert(
          'targetIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        options: $checkedConvert('options', (v) => v as Map<String, dynamic>?),
      );
      return val;
    });

Map<String, dynamic> _$TaskCreateFormToJson(TaskCreateForm instance) =>
    <String, dynamic>{
      'type': instance.type,
      'targetId': ?instance.targetId,
      'targetIds': ?instance.targetIds,
      'options': ?instance.options,
    };
