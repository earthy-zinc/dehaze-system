import 'package:json_annotation/json_annotation.dart';

part 'user_model.g.dart';

/// 用户信息
@JsonSerializable()
class UserModel {
  const UserModel({
    required this.userId,
    required this.username,
    this.nickname,
    this.avatar,
    this.deptId,
    this.deptName,
    this.roles = const [],
    this.perms = const [],
    this.dataScope,
    this.status,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) =>
      _$UserModelFromJson(json);

  /// 用户 ID
  @JsonKey(name: 'userId')
  final int userId;

  /// 用户名
  final String username;

  /// 昵称（后端 /auth/me 可能不返回，为空时回退用用户名）
  final String? nickname;

  /// 头像 URL
  final String? avatar;

  /// 部门 ID
  @JsonKey(name: 'deptId')
  final int? deptId;

  /// 部门名称
  @JsonKey(name: 'deptName')
  final String? deptName;

  /// 角色编码列表（如 ["ROOT", "ADMIN"]）
  final List<String> roles;

  /// 权限标识列表（如 ["sys:user:add", "sys:dataset:edit"]）
  final List<String> perms;

  /// 数据权限范围（1=全部 2=自定义 3=本部门 4=本部门及以下 5=仅本人）
  @JsonKey(name: 'dataScope')
  final int? dataScope;

  /// 状态（1=启用 0=禁用）
  final int? status;

  Map<String, dynamic> toJson() => _$UserModelToJson(this);

  /// 是否为 ROOT 角色
  bool get isRoot => roles.contains('ROOT');

  /// 是否已登录（有 userId 且有角色）
  bool get isAuthenticated => userId > 0;

  /// 检查是否拥有指定权限
  bool hasPermission(String perm) {
    if (isRoot) return true;
    return perms.contains(perm);
  }

  /// 检查是否拥有任一权限
  bool hasAnyPermission(List<String> permList) {
    if (isRoot) return true;
    return permList.any(perms.contains);
  }

  /// 检查是否拥有指定角色
  bool hasRole(String role) => roles.contains(role);

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
