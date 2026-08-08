// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'user_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

UserModel _$UserModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('UserModel', json, ($checkedConvert) {
      final val = UserModel(
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
        phone: $checkedConvert('phone', (v) => v as String?),
        deptId: $checkedConvert('deptId', (v) => (v as num?)?.toInt()),
        deptName: $checkedConvert('deptName', (v) => v as String?),
        roleIds: $checkedConvert(
          'roleIds',
          (v) =>
              (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList() ??
              [],
        ),
        roleNames: $checkedConvert(
          'roleNames',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        permissions: $checkedConvert(
          'permissions',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$UserModelToJson(UserModel instance) => <String, dynamic>{
  'userId': instance.userId,
  'username': instance.username,
  if (instance.nickname case final value?) 'nickname': value,
  if (instance.avatar case final value?) 'avatar': value,
  if (instance.email case final value?) 'email': value,
  if (instance.phone case final value?) 'phone': value,
  if (instance.deptId case final value?) 'deptId': value,
  if (instance.deptName case final value?) 'deptName': value,
  'roleIds': instance.roleIds,
  'roleNames': instance.roleNames,
  if (instance.dataScope case final value?) 'dataScope': value,
  if (instance.status case final value?) 'status': value,
  'permissions': instance.permissions,
  if (instance.createTime case final value?) 'createTime': value,
};

UserDetail _$UserDetailFromJson(Map<String, dynamic> json) =>
    $checkedCreate('UserDetail', json, ($checkedConvert) {
      final val = UserDetail(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
        phone: $checkedConvert('phone', (v) => v as String?),
        gender: $checkedConvert('gender', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        deptId: $checkedConvert('deptId', (v) => (v as num?)?.toInt()),
        deptName: $checkedConvert('deptName', (v) => v as String?),
        roleIds: $checkedConvert(
          'roleIds',
          (v) =>
              (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList() ??
              [],
        ),
        roleNames: $checkedConvert(
          'roleNames',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$UserDetailToJson(UserDetail instance) =>
    <String, dynamic>{
      'id': instance.id,
      'username': instance.username,
      if (instance.nickname case final value?) 'nickname': value,
      if (instance.avatar case final value?) 'avatar': value,
      if (instance.email case final value?) 'email': value,
      if (instance.phone case final value?) 'phone': value,
      if (instance.gender case final value?) 'gender': value,
      if (instance.status case final value?) 'status': value,
      if (instance.deptId case final value?) 'deptId': value,
      if (instance.deptName case final value?) 'deptName': value,
      'roleIds': instance.roleIds,
      'roleNames': instance.roleNames,
      if (instance.createTime case final value?) 'createTime': value,
      if (instance.updateTime case final value?) 'updateTime': value,
    };

UserQuery _$UserQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'UserQuery',
  json,
  ($checkedConvert) {
    final val = UserQuery(
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      keywords: $checkedConvert('keywords', (v) => v as String?),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      deptId: $checkedConvert('deptId', (v) => (v as num?)?.toInt()),
      startTime: $checkedConvert('startTime', (v) => v as String?),
      endTime: $checkedConvert('endTime', (v) => v as String?),
    );
    return val;
  },
);

Map<String, dynamic> _$UserQueryToJson(UserQuery instance) => <String, dynamic>{
  'pageNum': instance.pageNum,
  'pageSize': instance.pageSize,
  if (instance.keywords case final value?) 'keywords': value,
  if (instance.status case final value?) 'status': value,
  if (instance.deptId case final value?) 'deptId': value,
  if (instance.startTime case final value?) 'startTime': value,
  if (instance.endTime case final value?) 'endTime': value,
};

UserPageVO _$UserPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('UserPageVO', json, ($checkedConvert) {
      final val = UserPageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String?),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
        phone: $checkedConvert('mobile', (v) => v as String?),
        genderLabel: $checkedConvert('genderLabel', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        deptName: $checkedConvert('deptName', (v) => v as String?),
        roleNames: $checkedConvert('roleNames', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    }, fieldKeyMap: const {'phone': 'mobile'});

Map<String, dynamic> _$UserPageVOToJson(UserPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.username case final value?) 'username': value,
      if (instance.nickname case final value?) 'nickname': value,
      if (instance.avatar case final value?) 'avatar': value,
      if (instance.email case final value?) 'email': value,
      if (instance.phone case final value?) 'mobile': value,
      if (instance.genderLabel case final value?) 'genderLabel': value,
      if (instance.status case final value?) 'status': value,
      if (instance.deptName case final value?) 'deptName': value,
      if (instance.roleNames case final value?) 'roleNames': value,
      if (instance.createTime case final value?) 'createTime': value,
    };

UserForm _$UserFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('UserForm', json, ($checkedConvert) {
      final val = UserForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        email: $checkedConvert('email', (v) => v as String?),
        phone: $checkedConvert('mobile', (v) => v as String?),
        password: $checkedConvert('password', (v) => v as String?),
        gender: $checkedConvert('gender', (v) => (v as num?)?.toInt()),
        deptId: $checkedConvert('deptId', (v) => (v as num?)?.toInt()),
        roleIds: $checkedConvert(
          'roleIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    }, fieldKeyMap: const {'phone': 'mobile'});

Map<String, dynamic> _$UserFormToJson(UserForm instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  'username': instance.username,
  if (instance.nickname case final value?) 'nickname': value,
  if (instance.avatar case final value?) 'avatar': value,
  if (instance.email case final value?) 'email': value,
  if (instance.phone case final value?) 'mobile': value,
  if (instance.password case final value?) 'password': value,
  if (instance.gender case final value?) 'gender': value,
  if (instance.deptId case final value?) 'deptId': value,
  if (instance.roleIds case final value?) 'roleIds': value,
  if (instance.status case final value?) 'status': value,
};
