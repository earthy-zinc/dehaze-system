// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'user_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

UserModel _$UserModelFromJson(Map<String, dynamic> json) => $checkedCreate(
  'UserModel',
  json,
  ($checkedConvert) {
    final val = UserModel(
      userId: $checkedConvert('userId', (v) => (v as num).toInt()),
      username: $checkedConvert('username', (v) => v as String),
      nickname: $checkedConvert('nickname', (v) => v as String?),
      avatar: $checkedConvert('avatar', (v) => v as String?),
      deptId: $checkedConvert('deptId', (v) => (v as num?)?.toInt()),
      deptName: $checkedConvert('deptName', (v) => v as String?),
      roles: $checkedConvert(
        'roles',
        (v) =>
            (v as List<dynamic>?)?.map((e) => e as String).toList() ?? const [],
      ),
      permissions: $checkedConvert(
        'permissions',
        (v) =>
            (v as List<dynamic>?)?.map((e) => e as String).toList() ?? const [],
      ),
      dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$UserModelToJson(UserModel instance) => <String, dynamic>{
  'userId': instance.userId,
  'username': instance.username,
  if (instance.nickname case final value?) 'nickname': value,
  if (instance.avatar case final value?) 'avatar': value,
  if (instance.deptId case final value?) 'deptId': value,
  if (instance.deptName case final value?) 'deptName': value,
  'roles': instance.roles,
  'permissions': instance.permissions,
  if (instance.dataScope case final value?) 'dataScope': value,
  if (instance.status case final value?) 'status': value,
};
