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
          'roles',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        dataScope: $checkedConvert('dataScope', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        permissions: $checkedConvert(
          'perms',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    }, fieldKeyMap: const {'roleNames': 'roles', 'permissions': 'perms'});

Map<String, dynamic> _$UserModelToJson(UserModel instance) => <String, dynamic>{
  'userId': instance.userId,
  'username': instance.username,
  'nickname': ?instance.nickname,
  'avatar': ?instance.avatar,
  'email': ?instance.email,
  'phone': ?instance.phone,
  'deptId': ?instance.deptId,
  'deptName': ?instance.deptName,
  'roleIds': instance.roleIds,
  'roles': instance.roleNames,
  'dataScope': ?instance.dataScope,
  'status': ?instance.status,
  'perms': instance.permissions,
  'createTime': ?instance.createTime,
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
      'nickname': ?instance.nickname,
      'avatar': ?instance.avatar,
      'email': ?instance.email,
      'phone': ?instance.phone,
      'gender': ?instance.gender,
      'status': ?instance.status,
      'deptId': ?instance.deptId,
      'deptName': ?instance.deptName,
      'roleIds': instance.roleIds,
      'roleNames': instance.roleNames,
      'createTime': ?instance.createTime,
      'updateTime': ?instance.updateTime,
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
  'keywords': ?instance.keywords,
  'status': ?instance.status,
  'deptId': ?instance.deptId,
  'startTime': ?instance.startTime,
  'endTime': ?instance.endTime,
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
      'username': ?instance.username,
      'nickname': ?instance.nickname,
      'avatar': ?instance.avatar,
      'email': ?instance.email,
      'mobile': ?instance.phone,
      'genderLabel': ?instance.genderLabel,
      'status': ?instance.status,
      'deptName': ?instance.deptName,
      'roleNames': ?instance.roleNames,
      'createTime': ?instance.createTime,
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
  'id': ?instance.id,
  'username': instance.username,
  'nickname': ?instance.nickname,
  'avatar': ?instance.avatar,
  'email': ?instance.email,
  'mobile': ?instance.phone,
  'password': ?instance.password,
  'gender': ?instance.gender,
  'deptId': ?instance.deptId,
  'roleIds': ?instance.roleIds,
  'status': ?instance.status,
};
