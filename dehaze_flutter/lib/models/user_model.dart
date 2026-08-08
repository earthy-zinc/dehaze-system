import 'package:json_annotation/json_annotation.dart';

part 'user_model.g.dart';

/// 当前登录用户信息（对应 JS SDK UserInfo / 后端 /auth/me 返回）
///
/// 对齐后端 AuthUserInfo 结构。
@JsonSerializable()
class UserModel {
  const UserModel({
    required this.userId,
    required this.username,
    this.nickname,
    this.avatar,
    this.email,
    this.phone,
    this.deptId,
    this.deptName,
    this.roleIds = const [],
    this.roleNames = const [],
    this.dataScope,
    this.status,
    this.permissions = const [],
    this.createTime,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) =>
      _$UserModelFromJson(json);

  /// 用户 ID
  final int userId;

  /// 用户名
  final String username;

  /// 昵称（后端 /auth/me 可能不返回，为空时回退用用户名）
  final String? nickname;

  /// 头像 URL
  final String? avatar;

  /// 邮箱
  final String? email;

  /// 手机号
  final String? phone;

  /// 部门 ID
  final int? deptId;

  /// 部门名称
  final String? deptName;

  /// 角色 ID 列表
  @JsonKey(defaultValue: [])
  final List<int> roleIds;

  /// 角色名称列表
  @JsonKey(defaultValue: [])
  final List<String> roleNames;

  /// 数据权限范围（1=全部 2=本部门 3=本部门及以下 4=仅本人 5=自定义）
  final int? dataScope;

  /// 状态（1=启用 0=禁用）
  final int? status;

  /// 权限标识列表（如 ["sys:user:add", "sys:dataset:edit"]）
  @JsonKey(defaultValue: [])
  final List<String> permissions;

  /// 创建时间
  final String? createTime;

  Map<String, dynamic> toJson() => _$UserModelToJson(this);

  /// 是否为 ROOT 角色
  bool get isRoot => roleNames.contains('ROOT') || roleNames.any((r) => r.toUpperCase() == 'ROOT');

  /// 是否已登录（有 userId 且有角色）
  bool get isAuthenticated => userId > 0;

  /// 检查是否拥有指定权限
  bool hasPermission(String perm) {
    if (isRoot) return true;
    return permissions.contains(perm);
  }

  /// 检查是否拥有任一权限
  bool hasAnyPermission(List<String> permList) {
    if (isRoot) return true;
    return permList.any(permissions.contains);
  }

  /// 检查是否拥有指定角色
  bool hasRole(String role) => roleNames.contains(role);

  /// 获取头像首字母（用于默认头像）
  String get avatarInitials {
    final nick = nickname;
    if (nick != null && nick.isNotEmpty) {
      return nick.substring(0, 1).toUpperCase();
    }
    if (username.isNotEmpty) {
      return username.substring(0, 1).toUpperCase();
    }
    return 'U';
  }
}

/// 管理端用户信息详情（对应 JS SDK 中管理端用户详情）
///
/// 对齐后端 UserDetailVO。
@JsonSerializable()
class UserDetail {
  const UserDetail({
    required this.id,
    required this.username,
    this.nickname,
    this.avatar,
    this.email,
    this.phone,
    this.gender,
    this.status,
    this.deptId,
    this.deptName,
    this.roleIds = const [],
    this.roleNames = const [],
    this.createTime,
    this.updateTime,
  });

  factory UserDetail.fromJson(Map<String, dynamic> json) =>
      _$UserDetailFromJson(json);

  final int id;
  final String username;
  final String? nickname;
  final String? avatar;
  final String? email;
  final String? phone;
  final int? gender;
  final int? status;
  final int? deptId;
  final String? deptName;

  @JsonKey(defaultValue: [])
  final List<int> roleIds;

  @JsonKey(defaultValue: [])
  final List<String> roleNames;

  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$UserDetailToJson(this);
}

/// 用户查询参数（对应 JS SDK UserQuery）
@JsonSerializable()
class UserQuery {
  const UserQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keywords,
    this.status,
    this.deptId,
    this.startTime,
    this.endTime,
  });

  factory UserQuery.fromJson(Map<String, dynamic> json) =>
      _$UserQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? keywords;
  final int? status;
  final int? deptId;
  final String? startTime;
  final String? endTime;

  Map<String, dynamic> toJson() => _$UserQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
        if (status != null) 'status': status,
        if (deptId != null) 'deptId': deptId,
        if (startTime != null && startTime!.isNotEmpty) 'startTime': startTime,
        if (endTime != null && endTime!.isNotEmpty) 'endTime': endTime,
      };
}

/// 用户分页项（对应 JS SDK UserPageVO）
@JsonSerializable()
class UserPageVO {
  const UserPageVO({
    required this.id,
    this.username,
    this.nickname,
    this.avatar,
    this.email,
    this.phone,
    this.genderLabel,
    this.status,
    this.deptName,
    this.roleNames,
    this.createTime,
  });

  factory UserPageVO.fromJson(Map<String, dynamic> json) =>
      _$UserPageVOFromJson(json);

  final int id;
  final String? username;
  final String? nickname;
  final String? avatar;
  final String? email;

  @JsonKey(name: 'mobile')
  final String? phone;

  final String? genderLabel;
  final int? status;
  final String? deptName;

  final String? roleNames;

  final String? createTime;

  Map<String, dynamic> toJson() => _$UserPageVOToJson(this);
}

/// 用户表单（对应 JS SDK UserForm）
@JsonSerializable()
class UserForm {
  const UserForm({
    this.id,
    required this.username,
    this.nickname,
    this.avatar,
    this.email,
    this.phone,
    this.password,
    this.gender,
    this.deptId,
    this.roleIds,
    this.status,
  });

  factory UserForm.fromJson(Map<String, dynamic> json) =>
      _$UserFormFromJson(json);

  final int? id;
  final String username;
  final String? nickname;
  final String? avatar;
  final String? email;

  @JsonKey(name: 'mobile')
  final String? phone;

  final String? password;
  final int? gender;
  final int? deptId;

  final List<int>? roleIds;

  final int? status;

  Map<String, dynamic> toJson() => _$UserFormToJson(this);
}
