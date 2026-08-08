// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'menu_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Menu _$MenuFromJson(Map<String, dynamic> json) =>
    $checkedCreate('Menu', json, ($checkedConvert) {
      final val = Menu(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => _menuTypeFromJson(v)),
        icon: $checkedConvert('icon', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        component: $checkedConvert('component', (v) => v as String?),
        perm: $checkedConvert('perm', (v) => v as String?),
        sort: $checkedConvert('sort', (v) => (v as num).toInt()),
        visible: $checkedConvert('visible', (v) => _visibleFromJson(v)),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => Menu.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        redirect: $checkedConvert('redirect', (v) => v as String?),
        alwaysShow: $checkedConvert('alwaysShow', (v) => (v as num?)?.toInt()),
        breadcrumb: $checkedConvert('breadcrumb', (v) => (v as num?)?.toInt()),
        meta: $checkedConvert('meta', (v) => v as Map<String, dynamic>?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$MenuToJson(Menu instance) => <String, dynamic>{
  'id': instance.id,
  'parentId': instance.parentId,
  'name': instance.name,
  if (_menuTypeToJson(instance.type) case final value?) 'type': value,
  if (instance.icon case final value?) 'icon': value,
  if (instance.path case final value?) 'path': value,
  if (instance.component case final value?) 'component': value,
  if (instance.perm case final value?) 'perm': value,
  'sort': instance.sort,
  if (_visibleToJson(instance.visible) case final value?) 'visible': value,
  'status': instance.status,
  if (instance.children?.map((e) => e.toJson()).toList() case final value?)
    'children': value,
  if (instance.redirect case final value?) 'redirect': value,
  if (instance.alwaysShow case final value?) 'alwaysShow': value,
  if (instance.breadcrumb case final value?) 'breadcrumb': value,
  if (instance.meta case final value?) 'meta': value,
  if (instance.createTime case final value?) 'createTime': value,
  if (instance.updateTime case final value?) 'updateTime': value,
};

MenuQuery _$MenuQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MenuQuery', json, ($checkedConvert) {
      final val = MenuQuery(
        name: $checkedConvert('name', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        visible: $checkedConvert('visible', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$MenuQueryToJson(MenuQuery instance) => <String, dynamic>{
  if (instance.name case final value?) 'name': value,
  if (instance.status case final value?) 'status': value,
  if (instance.visible case final value?) 'visible': value,
};

MenuForm _$MenuFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MenuForm', json, ($checkedConvert) {
      final val = MenuForm(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => _menuTypeFromJson(v)),
        icon: $checkedConvert('icon', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        component: $checkedConvert('component', (v) => v as String?),
        perm: $checkedConvert('perm', (v) => v as String?),
        sort: $checkedConvert('sort', (v) => (v as num).toInt()),
        visible: $checkedConvert('visible', (v) => _visibleFromJson(v)),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        redirect: $checkedConvert('redirect', (v) => v as String?),
        alwaysShow: $checkedConvert('alwaysShow', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$MenuFormToJson(MenuForm instance) => <String, dynamic>{
  if (instance.id case final value?) 'id': value,
  'parentId': instance.parentId,
  'name': instance.name,
  if (_menuTypeToJson(instance.type) case final value?) 'type': value,
  if (instance.icon case final value?) 'icon': value,
  if (instance.path case final value?) 'path': value,
  if (instance.component case final value?) 'component': value,
  if (instance.perm case final value?) 'perm': value,
  'sort': instance.sort,
  if (_visibleToJson(instance.visible) case final value?) 'visible': value,
  'status': instance.status,
  if (instance.redirect case final value?) 'redirect': value,
  if (instance.alwaysShow case final value?) 'alwaysShow': value,
};

MenuOption _$MenuOptionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MenuOption', json, ($checkedConvert) {
      final val = MenuOption(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => MenuOption.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$MenuOptionToJson(MenuOption instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      if (instance.children?.map((e) => e.toJson()).toList() case final value?)
        'children': value,
    };

RouteMeta _$RouteMetaFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RouteMeta', json, ($checkedConvert) {
      final val = RouteMeta(
        title: $checkedConvert('title', (v) => v as String?),
        icon: $checkedConvert('icon', (v) => v as String?),
        hidden: $checkedConvert('hidden', (v) => v as bool?),
        keepAlive: $checkedConvert('keepAlive', (v) => v as bool?),
        alwaysShow: $checkedConvert('alwaysShow', (v) => v as bool?),
        breadcrumb: $checkedConvert('breadcrumb', (v) => v as bool?),
        noCache: $checkedConvert('noCache', (v) => v as bool?),
        affix: $checkedConvert('affix', (v) => v as bool?),
        activeMenu: $checkedConvert('activeMenu', (v) => v as String?),
        params: $checkedConvert('params', (v) => v as Map<String, dynamic>?),
      );
      return val;
    });

Map<String, dynamic> _$RouteMetaToJson(RouteMeta instance) => <String, dynamic>{
  if (instance.title case final value?) 'title': value,
  if (instance.icon case final value?) 'icon': value,
  if (instance.hidden case final value?) 'hidden': value,
  if (instance.keepAlive case final value?) 'keepAlive': value,
  if (instance.alwaysShow case final value?) 'alwaysShow': value,
  if (instance.breadcrumb case final value?) 'breadcrumb': value,
  if (instance.noCache case final value?) 'noCache': value,
  if (instance.affix case final value?) 'affix': value,
  if (instance.activeMenu case final value?) 'activeMenu': value,
  if (instance.params case final value?) 'params': value,
};

RouteVO _$RouteVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RouteVO', json, ($checkedConvert) {
      final val = RouteVO(
        name: $checkedConvert('name', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        component: $checkedConvert('component', (v) => v as String?),
        redirect: $checkedConvert('redirect', (v) => v as String?),
        meta: $checkedConvert(
          'meta',
          (v) =>
              v == null ? null : RouteMeta.fromJson(v as Map<String, dynamic>),
        ),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => RouteVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$RouteVOToJson(RouteVO instance) => <String, dynamic>{
  if (instance.name case final value?) 'name': value,
  if (instance.path case final value?) 'path': value,
  if (instance.component case final value?) 'component': value,
  if (instance.redirect case final value?) 'redirect': value,
  if (instance.meta?.toJson() case final value?) 'meta': value,
  if (instance.children?.map((e) => e.toJson()).toList() case final value?)
    'children': value,
};
