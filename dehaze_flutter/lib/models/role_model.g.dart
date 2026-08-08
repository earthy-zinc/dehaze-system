// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'role_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Role _$RoleFromJson(Map<String, dynamic> json) => $checkedCreate('Role', json, (
  $checkedConvert,
) {
  final val = Role(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    code: $checkedConvert('code', (v) => v as String),
    sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
    status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
    dataScopeName: $checkedConvert('dataScopeName', (v) => v as String?),
    menuIds: $checkedConvert(
      'menuIds',
      (v) =>
          (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList() ?? [],
    ),
    deptIds: $checkedConvert(
      'deptIds',
      (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
    ),
    remark: $checkedConvert('remark', (v) => v as String?),
    createTime: $checkedConvert('createTime', (v) => v as String?),
    updateTime: $checkedConvert('updateTime', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$RoleToJson(Role instance) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'code': instance.code,
  if (instance.sort case final value?) 'sort': value,
  if (instance.status case final value?) 'status': value,
  if (instance.dataScope case final value?) 'dataScope': value,
  if (instance.dataScopeName case final value?) 'dataScopeName': value,
  'menuIds': instance.menuIds,
  if (instance.deptIds case final value?) 'deptIds': value,
  if (instance.remark case final value?) 'remark': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

RoleQuery _$RoleQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'RoleQuery',
  json,
  ($checkedConvert) {
    final val = RoleQuery(
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      keywords: $checkedConvert('keywords', (v) => v as String?),
    );
    return val;
  },
);

Map<String, dynamic> _$RoleQueryToJson(RoleQuery instance) => <String, dynamic>{
  'pageNum': instance.pageNum,
  'pageSize': instance.pageSize,
  if (instance.keywords case final value?) 'keywords': value,
};

RolePageVO _$RolePageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RolePageVO', json, ($checkedConvert) {
      final val = RolePageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String?),
        code: $checkedConvert('code', (v) => v as String?),
        sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RolePageVOToJson(RolePageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.name case final value?) 'name': value,
      if (instance.code case final value?) 'code': value,
      if (instance.sort case final value?) 'sort': value,
      if (instance.status case final value?) 'status': value,
      if (instance.dataScope case final value?) 'dataScope': value,
      if (instance.createTime case final value?) 'createTime': value,
    };

RoleForm _$RoleFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RoleForm', json, ($checkedConvert) {
      final val = RoleForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        name: $checkedConvert('name', (v) => v as String),
        code: $checkedConvert('code', (v) => v as String),
        sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
        menuIds: $checkedConvert(
          'menuIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        deptIds: $checkedConvert(
          'deptIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        remark: $checkedConvert('remark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RoleFormToJson(RoleForm instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  'name': instance.name,
  'code': instance.code,
  if (instance.sort case final value?) 'sort': value,
  if (instance.status case final value?) 'status': value,
  if (instance.dataScope case final value?) 'dataScope': value,
  if (instance.menuIds case final value?) 'menuIds': value,
  if (instance.deptIds case final value?) 'deptIds': value,
  if (instance.remark case final value?) 'remark': value,
};

RoleOption _$RoleOptionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RoleOption', json, ($checkedConvert) {
      final val = RoleOption(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        code: $checkedConvert('code', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$RoleOptionToJson(RoleOption instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'code': instance.code,
    };
