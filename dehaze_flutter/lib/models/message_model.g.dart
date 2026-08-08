// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'message_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

MessageVO _$MessageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MessageVO', json, ($checkedConvert) {
      final val = MessageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        type: $checkedConvert('type', (v) => v as String),
        typeLabel: $checkedConvert('typeLabel', (v) => v as String),
        title: $checkedConvert('title', (v) => v as String),
        summary: $checkedConvert('summary', (v) => v as String?),
        content: $checkedConvert('content', (v) => v as String?),
        priority: $checkedConvert('priority', (v) => (v as num).toInt()),
        readStatus: $checkedConvert('readStatus', (v) => (v as num).toInt()),
        senderType: $checkedConvert('senderType', (v) => (v as num).toInt()),
        readTime: $checkedConvert('readTime', (v) => v as String?),
        jumpUrl: $checkedConvert('jumpUrl', (v) => v as String?),
        extra: $checkedConvert('extra', (v) => v as Map<String, dynamic>?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$MessageVOToJson(MessageVO instance) => <String, dynamic>{
  'id': instance.id,
  'type': instance.type,
  'typeLabel': instance.typeLabel,
  'title': instance.title,
  if (instance.summary case final value?) 'summary': value,
  if (instance.content case final value?) 'content': value,
  'priority': instance.priority,
  'readStatus': instance.readStatus,
  'senderType': instance.senderType,
  if (instance.readTime case final value?) 'readTime': value,
  if (instance.jumpUrl case final value?) 'jumpUrl': value,
  if (instance.extra case final value?) 'extra': value,
  'createTime': instance.createTime,
};

UnreadCountVO _$UnreadCountVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('UnreadCountVO', json, ($checkedConvert) {
      final val = UnreadCountVO(
        count: $checkedConvert('count', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$UnreadCountVOToJson(UnreadCountVO instance) =>
    <String, dynamic>{'count': instance.count};

ReadAllResult _$ReadAllResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ReadAllResult', json, ($checkedConvert) {
      final val = ReadAllResult(
        affectedCount: $checkedConvert(
          'affectedCount',
          (v) => (v as num).toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ReadAllResultToJson(ReadAllResult instance) =>
    <String, dynamic>{'affectedCount': instance.affectedCount};
