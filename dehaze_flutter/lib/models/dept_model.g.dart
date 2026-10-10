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
  'leader': ?instance.leader,
  'phone': ?instance.phone,
  'email': ?instance.email,
  'children': ?instance.children?.map((e) => e.toJson()).toList(),
  'createTime': ?instance.createTime,
  'updateTime': ?instance.updateTime,
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
  'name': ?instance.name,
  'status': ?instance.status,
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
  'id': ?instance.id,
  'parentId': instance.parentId,
  'name': instance.name,
  'sort': instance.sort,
  'status': instance.status,
  'leader': ?instance.leader,
  'phone': ?instance.phone,
  'email': ?instance.email,
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
      'children': ?instance.children?.map((e) => e.toJson()).toList(),
    };
