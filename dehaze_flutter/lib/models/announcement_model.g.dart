// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'announcement_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AnnouncementVO _$AnnouncementVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AnnouncementVO', json, ($checkedConvert) {
  final val = AnnouncementVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    title: $checkedConvert('title', (v) => v as String),
    type: $checkedConvert('type', (v) => v as String),
    importance: $checkedConvert('importance', (v) => (v as num).toInt()),
    targetScope: $checkedConvert('targetScope', (v) => _targetScopeFromJson(v)),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
    content: $checkedConvert('content', (v) => v as String?),
    typeLabel: $checkedConvert('typeLabel', (v) => v as String?),
    importanceLabel: $checkedConvert('importanceLabel', (v) => v as String?),
    targetScopeLabel: $checkedConvert('targetScopeLabel', (v) => v as String?),
    targetParams: $checkedConvert(
      'targetParams',
      (v) => v as Map<String, dynamic>?,
    ),
    statusLabel: $checkedConvert('statusLabel', (v) => v as String?),
    sendTime: $checkedConvert('sendTime', (v) => v as String?),
    expireTime: $checkedConvert('expireTime', (v) => v as String?),
    sentCount: $checkedConvert('sentCount', (v) => (v as num?)?.toInt()),
    updateTime: $checkedConvert('updateTime', (v) => v as String?),
    createBy: $checkedConvert('createBy', (v) => (v as num?)?.toInt()),
  );
  return val;
});

Map<String, dynamic> _$AnnouncementVOToJson(AnnouncementVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'title': instance.title,
      'content': ?instance.content,
      'type': instance.type,
      'typeLabel': ?instance.typeLabel,
      'importance': instance.importance,
      'importanceLabel': ?instance.importanceLabel,
      'targetScope': _targetScopeToJson(instance.targetScope),
      'targetScopeLabel': ?instance.targetScopeLabel,
      'targetParams': ?instance.targetParams,
      'status': instance.status,
      'statusLabel': ?instance.statusLabel,
      'sendTime': ?instance.sendTime,
      'expireTime': ?instance.expireTime,
      'sentCount': ?instance.sentCount,
      'createTime': instance.createTime,
      'updateTime': ?instance.updateTime,
      'createBy': ?instance.createBy,
    };

AnnouncementQuery _$AnnouncementQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnnouncementQuery', json, ($checkedConvert) {
      final val = AnnouncementQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        title: $checkedConvert('title', (v) => v as String?),
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$AnnouncementQueryToJson(AnnouncementQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'title': ?instance.title,
      'type': ?instance.type,
      'status': ?instance.status,
    };

AnnouncementForm _$AnnouncementFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnnouncementForm', json, ($checkedConvert) {
      final val = AnnouncementForm(
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        importance: $checkedConvert('importance', (v) => (v as num).toInt()),
        targetScope: $checkedConvert(
          'targetScope',
          (v) => _targetScopeFromJson(v),
        ),
        targetParams: $checkedConvert(
          'targetParams',
          (v) => v as Map<String, dynamic>?,
        ),
        sendTime: $checkedConvert('sendTime', (v) => v as String?),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AnnouncementFormToJson(AnnouncementForm instance) =>
    <String, dynamic>{
      'title': instance.title,
      'content': instance.content,
      'type': instance.type,
      'importance': instance.importance,
      'targetScope': _targetScopeToJson(instance.targetScope),
      'targetParams': ?instance.targetParams,
      'sendTime': ?instance.sendTime,
      'expireTime': ?instance.expireTime,
    };

AnnouncementSendResult _$AnnouncementSendResultFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AnnouncementSendResult', json, ($checkedConvert) {
  final val = AnnouncementSendResult(
    sentCount: $checkedConvert('sentCount', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$AnnouncementSendResultToJson(
  AnnouncementSendResult instance,
) => <String, dynamic>{'sentCount': instance.sentCount};
