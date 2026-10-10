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
      'typeName': ?instance.typeName,
      'variables': ?instance.variables,
      'status': instance.status,
      'statusName': ?instance.statusName,
      'description': ?instance.description,
      'createTime': instance.createTime,
      'updateTime': ?instance.updateTime,
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
  'code': ?instance.code,
  'name': ?instance.name,
  'type': ?instance.type,
  'status': ?instance.status,
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
  'id': ?instance.id,
  'code': instance.code,
  'name': instance.name,
  'title': instance.title,
  'content': instance.content,
  'type': instance.type,
  'variables': ?instance.variables,
  'status': instance.status,
  'description': ?instance.description,
};
