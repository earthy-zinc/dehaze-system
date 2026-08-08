import 'package:json_annotation/json_annotation.dart';

part 'role_model.g.dart';

/// 数据权限范围枚举
///
/// 对应后端 DataScopeEnum。
enum DataScope {
  /// 全部数据权限
  @JsonValue(1)
  all,

  /// 本部门数据权限
  @JsonValue(2)
  dept,

  /// 本部门及以下数据权限
  @JsonValue(3)
  deptAndSub,

  /// 仅本人数据权限
  @JsonValue(4)
  self,

  /// 自定义数据权限
  @JsonValue(5)
  custom;

  int get value => switch (this) {
        DataScope.all => 1,
        DataScope.dept => 2,
        DataScope.deptAndSub => 3,
        DataScope.self => 4,
        DataScope.custom => 5,
      };

  static DataScope fromValue(int? value) {
    switch (value) {
      case 2:
        return DataScope.dept;
      case 3:
        return DataScope.deptAndSub;
      case 4:
        return DataScope.self;
      case 5:
        return DataScope.custom;
      default:
        return DataScope.all;
    }
  }

  String get label => switch (this) {
        DataScope.all => '全部数据权限',
        DataScope.dept => '本部门数据权限',
        DataScope.deptAndSub => '本部门及以下数据权限',
        DataScope.self => '仅本人数据权限',
        DataScope.custom => '自定义数据权限',
      };
}

/// 角色详情（对应 JS SDK Role）
@JsonSerializable()
class Role {
  const Role({
    required this.id,
    required this.name,
    required this.code,
    this.sort,
    this.status,
    this.dataScope,
    this.dataScopeName,
    this.menuIds = const [],
    this.deptIds,
    this.remark,
    this.createTime,
    this.updateTime,
  });

  factory Role.fromJson(Map<String, dynamic> json) => _$RoleFromJson(json);

  final int id;
  final String name;
  final String code;
  final int? sort;
  final int? status;
  final int? dataScope;
  final String? dataScopeName;

  @JsonKey(defaultValue: [])
  final List<int> menuIds;

  final List<int>? deptIds;

  final String? remark;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$RoleToJson(this);
}

/// 角色查询参数（对应 JS SDK RoleQuery）
@JsonSerializable()
class RoleQuery {
  const RoleQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keywords,
  });

  factory RoleQuery.fromJson(Map<String, dynamic> json) =>
      _$RoleQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? keywords;

  Map<String, dynamic> toJson() => _$RoleQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
      };
}

/// 角色分页项（对应 JS SDK RolePageVO）
@JsonSerializable()
class RolePageVO {
  const RolePageVO({
    required this.id,
    this.name,
    this.code,
    this.sort,
    this.status,
    this.dataScope,
    this.createTime,
  });

  factory RolePageVO.fromJson(Map<String, dynamic> json) =>
      _$RolePageVOFromJson(json);

  final int id;
  final String? name;
  final String? code;
  final int? sort;
  final int? status;
  final int? dataScope;
  final String? createTime;

  Map<String, dynamic> toJson() => _$RolePageVOToJson(this);
}

/// 角色表单（对应 JS SDK RoleForm）
@JsonSerializable()
class RoleForm {
  const RoleForm({
    this.id,
    required this.name,
    required this.code,
    this.sort,
    this.status,
    this.dataScope,
    this.menuIds,
    this.deptIds,
    this.remark,
  });

  factory RoleForm.fromJson(Map<String, dynamic> json) =>
      _$RoleFormFromJson(json);

  final int? id;
  final String name;
  final String code;
  final int? sort;
  final int? status;
  final int? dataScope;

  final List<int>? menuIds;

  final List<int>? deptIds;

  final String? remark;

  Map<String, dynamic> toJson() => _$RoleFormToJson(this);
}

/// 角色下拉选项（对应 JS SDK OptionType）
@JsonSerializable()
class RoleOption {
  const RoleOption({
    required this.id,
    required this.name,
    required this.code,
  });

  factory RoleOption.fromJson(Map<String, dynamic> json) =>
      _$RoleOptionFromJson(json);

  final int id;
  final String name;
  final String code;

  Map<String, dynamic> toJson() => _$RoleOptionToJson(this);
}
