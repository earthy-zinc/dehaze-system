// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dept_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Dept _$DeptFromJson(Map<String, dynamic> json) =>
    $checkedCreate('Dept', json, ($checkedConvert) {
      final val = Dept(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        sort: $checkedConvert('sort', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        leader: $checkedConvert('leader', (v) => v as String?),
        phone: $checkedConvert('phone', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => Dept.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DeptToJson(Dept instance) => <String, dynamic>{
  'id': instance.id,
  'parentId': instance.parentId,
  'name': instance.name,
  'sort': instance.sort,
  'status': instance.status,
  if (instance.leader case final value?) 'leader': value,
  if (instance.phone case final value?) 'phone': value,
  if (instance.email case final value?) 'email': value,
  if (instance.children?.map((e) => e.toJson()).toList() case final value?)
    'children': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

DeptQuery _$DeptQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DeptQuery', json, ($checkedConvert) {
      final val = DeptQuery(
        name: $checkedConvert('name', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$DeptQueryToJson(DeptQuery instance) => <String, dynamic>{
  if (instance.name case final value?) 'name': value,
  if (instance.status case final value?) 'status': value,
};

DeptForm _$DeptFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DeptForm', json, ($checkedConvert) {
      final val = DeptForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        sort: $checkedConvert('sort', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        leader: $checkedConvert('leader', (v) => v as String?),
        phone: $checkedConvert('phone', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DeptFormToJson(DeptForm instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  'parentId': instance.parentId,
  'name': instance.name,
  'sort': instance.sort,
  'status': instance.status,
  if (instance.leader case final value?) 'leader': value,
  if (instance.phone case final value?) 'phone': value,
  if (instance.email case final value?) 'email': value,
};

DeptOption _$DeptOptionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DeptOption', json, ($checkedConvert) {
      final val = DeptOption(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => DeptOption.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$DeptOptionToJson(DeptOption instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      if (instance.children?.map((e) => e.toJson()).toList() case final value?)
        'children': value,
    };
