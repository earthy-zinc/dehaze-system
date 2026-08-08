import 'package:json_annotation/json_annotation.dart';

part 'menu_model.g.dart';

// ==================== 枚举 ====================

/// 菜单类型：1=目录 2=菜单 3=按钮
enum MenuType {
  @JsonValue(1)
  directory,
  @JsonValue(2)
  menu,
  @JsonValue(3)
  button,
}

/// 菜单可见性：1=显示 0=隐藏
enum MenuVisible {
  @JsonValue(1)
  visible,
  @JsonValue(0)
  hidden,
}

// ==================== Menu ====================

@JsonSerializable()
class Menu {
  const Menu({
    required this.id,
    required this.parentId,
    required this.name,
    required this.type,
    this.icon,
    this.path,
    this.component,
    this.perm,
    required this.sort,
    required this.visible,
    required this.status,
    this.children,
    this.redirect,
    this.alwaysShow,
    this.breadcrumb,
    this.meta,
    this.createTime,
    this.updateTime,
  });

  factory Menu.fromJson(Map<String, dynamic> json) => _$MenuFromJson(json);

  final int id;

  @JsonKey(name: 'parentId')
  final int parentId;

  final String name;

  @JsonKey(fromJson: _menuTypeFromJson, toJson: _menuTypeToJson)
  final int type;

  final String? icon;
  final String? path;
  final String? component;
  final String? perm;
  final int sort;

  @JsonKey(fromJson: _visibleFromJson, toJson: _visibleToJson)
  final int visible;

  final int status;
  final List<Menu>? children;
  final String? redirect;

  @JsonKey(name: 'alwaysShow')
  final int? alwaysShow;

  final int? breadcrumb;
  final Map<String, dynamic>? meta;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$MenuToJson(this);
}

// ==================== MenuQuery ====================

@JsonSerializable()
class MenuQuery {
  const MenuQuery({this.name, this.status, this.visible});

  factory MenuQuery.fromJson(Map<String, dynamic> json) =>
      _$MenuQueryFromJson(json);

  final String? name;
  final int? status;
  final int? visible;

  Map<String, dynamic> toJson() => _$MenuQueryToJson(this);

  Map<String, dynamic> toQuery() => {
        if (name != null) 'keywords': name,
        if (status != null) 'status': status,
        if (visible != null) 'visible': visible,
      };
}

// ==================== MenuForm ====================

@JsonSerializable()
class MenuForm {
  const MenuForm({
    this.id,
    required this.parentId,
    required this.name,
    required this.type,
    this.icon,
    this.path,
    this.component,
    this.perm,
    required this.sort,
    required this.visible,
    required this.status,
    this.redirect,
    this.alwaysShow,
  });

  factory MenuForm.fromJson(Map<String, dynamic> json) =>
      _$MenuFormFromJson(json);

  final int? id;

  @JsonKey(name: 'parentId')
  final int parentId;

  final String name;

  @JsonKey(fromJson: _menuTypeFromJson, toJson: _menuTypeToJson)
  final int type;

  final String? icon;
  final String? path;
  final String? component;
  final String? perm;
  final int sort;

  @JsonKey(fromJson: _visibleFromJson, toJson: _visibleToJson)
  final int visible;

  final int status;
  final String? redirect;

  @JsonKey(name: 'alwaysShow')
  final int? alwaysShow;

  Map<String, dynamic> toJson() => _$MenuFormToJson(this);
}

// ==================== MenuOption ====================

@JsonSerializable()
class MenuOption {
  const MenuOption({
    required this.id,
    required this.name,
    this.children,
  });

  factory MenuOption.fromJson(Map<String, dynamic> json) =>
      _$MenuOptionFromJson(json);

  final int id;
  final String name;
  final List<MenuOption>? children;

  Map<String, dynamic> toJson() => _$MenuOptionToJson(this);
}

// ==================== RouteVO ====================

@JsonSerializable()
class RouteMeta {
  const RouteMeta({
    this.title,
    this.icon,
    this.hidden,
    this.keepAlive,
    this.alwaysShow,
    this.breadcrumb,
    this.noCache,
    this.affix,
    this.activeMenu,
    this.params,
  });

  factory RouteMeta.fromJson(Map<String, dynamic> json) =>
      _$RouteMetaFromJson(json);

  final String? title;
  final String? icon;
  final bool? hidden;

  @JsonKey(name: 'keepAlive')
  final bool? keepAlive;

  @JsonKey(name: 'alwaysShow')
  final bool? alwaysShow;

  final bool? breadcrumb;

  @JsonKey(name: 'noCache')
  final bool? noCache;

  final bool? affix;

  @JsonKey(name: 'activeMenu')
  final String? activeMenu;

  final Map<String, dynamic>? params;

  Map<String, dynamic> toJson() => _$RouteMetaToJson(this);
}

@JsonSerializable()
class RouteVO {
  const RouteVO({
    this.name,
    this.path,
    this.component,
    this.redirect,
    this.meta,
    this.children,
  });

  factory RouteVO.fromJson(Map<String, dynamic> json) =>
      _$RouteVOFromJson(json);

  final String? name;
  final String? path;
  final String? component;
  final String? redirect;
  final RouteMeta? meta;
  final List<RouteVO>? children;

  Map<String, dynamic> toJson() => _$RouteVOToJson(this);
}

// ==================== JSON 转换辅助函数 ====================

int _menuTypeFromJson(dynamic value) {
  if (value is int) return value;
  if (value is String) return int.tryParse(value) ?? 1;
  return 1;
}

dynamic _menuTypeToJson(int value) => value;

int _visibleFromJson(dynamic value) {
  if (value is int) return value;
  if (value is String) return int.tryParse(value) ?? 1;
  return 1;
}

dynamic _visibleToJson(int value) => value;
