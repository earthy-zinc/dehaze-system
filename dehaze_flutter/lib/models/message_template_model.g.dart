// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'message_template_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

MessageTemplateVO _$MessageTemplateVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MessageTemplateVO', json, ($checkedConvert) {
      final val = MessageTemplateVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        code: $checkedConvert('code', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        typeName: $checkedConvert('typeName', (v) => v as String?),
        variables: $checkedConvert(
          'variables',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        status: $checkedConvert('status', (v) => v as String),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$MessageTemplateVOToJson(MessageTemplateVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'code': instance.code,
      'name': instance.name,
      'title': instance.title,
      'content': instance.content,
      'type': instance.type,
      if (instance.typeName case final value?) 'typeName': value,
      if (instance.variables case final value?) 'variables': value,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      if (instance.description case final value?) 'description': value,
      'createTime': instance.createTime,
      if (instance.updateTime case final value?) 'updateTime': value,
    };

MessageTemplateQuery _$MessageTemplateQueryFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('MessageTemplateQuery', json, ($checkedConvert) {
  final val = MessageTemplateQuery(
    pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
    pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
    code: $checkedConvert('code', (v) => v as String?),
    name: $checkedConvert('name', (v) => v as String?),
    type: $checkedConvert('type', (v) => v as String?),
    status: $checkedConvert('status', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$MessageTemplateQueryToJson(
  MessageTemplateQuery instance,
) => <String, dynamic>{
  'pageNum': instance.pageNum,
  'pageSize': instance.pageSize,
  if (instance.code case final value?) 'code': value,
  if (instance.name case final value?) 'name': value,
  if (instance.type case final value?) 'type': value,
  if (instance.status case final value?) 'status': value,
};

MessageTemplateForm _$MessageTemplateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MessageTemplateForm', json, ($checkedConvert) {
      final val = MessageTemplateForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        code: $checkedConvert('code', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        variables: $checkedConvert(
          'variables',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        status: $checkedConvert('status', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$MessageTemplateFormToJson(
  MessageTemplateForm instance,
) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  'code': instance.code,
  'name': instance.name,
  'title': instance.title,
  'content': instance.content,
  'type': instance.type,
  if (instance.variables case final value?) 'variables': value,
  'status': instance.status,
  if (instance.description case final value?) 'description': value,
};
