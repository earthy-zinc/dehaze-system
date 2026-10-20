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
  'sort': ?instance.sort,
  'status': ?instance.status,
  'dataScope': ?instance.dataScope,
  'dataScopeName': ?instance.dataScopeName,
  'menuIds': instance.menuIds,
  'deptIds': ?instance.deptIds,
  'remark': ?instance.remark,
  'createTime': ?instance.createTime,
  'updateTime': ?instance.updateTime,
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
  'keywords': ?instance.keywords,
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
      'name': ?instance.name,
      'code': ?instance.code,
      'sort': ?instance.sort,
      'status': ?instance.status,
      'dataScope': ?instance.dataScope,
      'createTime': ?instance.createTime,
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
  'id': ?instance.id,
  'name': instance.name,
  'code': instance.code,
  'sort': ?instance.sort,
  'status': ?instance.status,
  'dataScope': ?instance.dataScope,
  'menuIds': ?instance.menuIds,
  'deptIds': ?instance.deptIds,
  'remark': ?instance.remark,
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
