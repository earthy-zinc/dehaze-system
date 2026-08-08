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
  if (_statusToJson(instance.status) case final value?) 'status': value,
  'progress': instance.progress,
  if (instance.taskType case final value?) 'taskType': value,
  if (_categoryToJson(instance.taskCategory) case final value?)
    'taskCategory': value,
  if (instance.totalFiles case final value?) 'totalFiles': value,
  if (instance.processedFiles case final value?) 'processedFiles': value,
  if (instance.downloadUrl case final value?) 'downloadUrl': value,
  if (instance.expiresAt case final value?) 'expiresAt': value,
  if (instance.createdAt case final value?) 'createdAt': value,
  if (instance.startedAt case final value?) 'startedAt': value,
  if (instance.completedAt case final value?) 'completedAt': value,
  if (instance.error case final value?) 'error': value,
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
      if (instance.targetId case final value?) 'targetId': value,
      if (instance.targetIds case final value?) 'targetIds': value,
      if (instance.options case final value?) 'options': value,
    };
