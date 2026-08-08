// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'announcement_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AnnouncementVO _$AnnouncementVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnnouncementVO', json, ($checkedConvert) {
      final val = AnnouncementVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String?),
        type: $checkedConvert('type', (v) => v as String),
        typeName: $checkedConvert('typeName', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        priority: $checkedConvert('priority', (v) => (v as num).toInt()),
        targetType: $checkedConvert('targetType', (v) => v as String),
        targetUsers: $checkedConvert(
          'targetUsers',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        sendTime: $checkedConvert('sendTime', (v) => v as String?),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
        createBy: $checkedConvert('createBy', (v) => (v as num?)?.toInt()),
        read: $checkedConvert('read', (v) => v as bool?),
      );
      return val;
    });

Map<String, dynamic> _$AnnouncementVOToJson(AnnouncementVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'title': instance.title,
      if (instance.content case final value?) 'content': value,
      'type': instance.type,
      if (instance.typeName case final value?) 'typeName': value,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      'priority': instance.priority,
      'targetType': instance.targetType,
      if (instance.targetUsers case final value?) 'targetUsers': value,
      if (instance.sendTime case final value?) 'sendTime': value,
      if (instance.expireTime case final value?) 'expireTime': value,
      'createTime': instance.createTime,
      if (instance.updateTime case final value?) 'updateTime': value,
      if (instance.createBy case final value?) 'createBy': value,
      if (instance.read case final value?) 'read': value,
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
        status: $checkedConvert('status', (v) => v as String?),
        startDate: $checkedConvert('startDate', (v) => v as String?),
        endDate: $checkedConvert('endDate', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AnnouncementQueryToJson(AnnouncementQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.title case final value?) 'title': value,
      if (instance.type case final value?) 'type': value,
      if (instance.status case final value?) 'status': value,
      if (instance.startDate case final value?) 'startDate': value,
      if (instance.endDate case final value?) 'endDate': value,
    };

AnnouncementForm _$AnnouncementFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AnnouncementForm', json, ($checkedConvert) {
      final val = AnnouncementForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        priority: $checkedConvert('priority', (v) => (v as num).toInt()),
        targetType: $checkedConvert('targetType', (v) => v as String),
        targetUsers: $checkedConvert(
          'targetUsers',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        sendNow: $checkedConvert('sendNow', (v) => v as bool?),
      );
      return val;
    });

Map<String, dynamic> _$AnnouncementFormToJson(AnnouncementForm instance) =>
    <String, dynamic>{
      if (instance.id case final value?) 'id': value,
      'title': instance.title,
      'content': instance.content,
      'type': instance.type,
      'priority': instance.priority,
      'targetType': instance.targetType,
      if (instance.targetUsers case final value?) 'targetUsers': value,
      if (instance.expireTime case final value?) 'expireTime': value,
      if (instance.sendNow case final value?) 'sendNow': value,
    };

AnnouncementSendResult _$AnnouncementSendResultFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('AnnouncementSendResult', json, ($checkedConvert) {
  final val = AnnouncementSendResult(
    successCount: $checkedConvert('successCount', (v) => (v as num).toInt()),
    failedCount: $checkedConvert('failedCount', (v) => (v as num).toInt()),
    totalCount: $checkedConvert('totalCount', (v) => (v as num).toInt()),
    sendTime: $checkedConvert('sendTime', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$AnnouncementSendResultToJson(
  AnnouncementSendResult instance,
) => <String, dynamic>{
  'successCount': instance.successCount,
  'failedCount': instance.failedCount,
  'totalCount': instance.totalCount,
  'sendTime': instance.sendTime,
};
