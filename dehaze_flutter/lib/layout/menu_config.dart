import 'package:flutter/material.dart';

/// 菜单项数据模型
class MenuItemData {
  const MenuItemData({
    required this.icon,
    required this.title,
    required this.route,
  });

  final IconData icon;
  final String title;
  final String route;
}

/// 菜单分组数据模型
class MenuSection {
  const MenuSection({
    required this.title,
    required this.items,
  });

  final String title;
  final List<MenuItemData> items;
}

/// 侧边菜单配置
/// 统一管理所有菜单项数据，便于维护和修改
class MenuConfig {
  const MenuConfig._();

  /// 获取所有菜单分组
  static const List<MenuSection> menuSections = [
    MenuSection(
      title: '处理流程',
      items: [
        MenuItemData(
          icon: Icons.image_outlined,
          title: '图像输入',
          route: '/image-input',
        ),
        MenuItemData(
          icon: Icons.psychology_outlined,
          title: '算法选择',
          route: '/algorithm-select',
        ),
        MenuItemData(
          icon: Icons.settings_outlined,
          title: '去雾处理',
          route: '/processing',
        ),
      ],
    ),
    MenuSection(
      title: '效果对比',
      items: [
        MenuItemData(
          icon: Icons.view_column_outlined,
          title: '并排对比',
          route: '/side-by-side',
        ),
        MenuItemData(
          icon: Icons.layers_outlined,
          title: '重叠对比',
          route: '/overlay',
        ),
        MenuItemData(
          icon: Icons.search_outlined,
          title: '放大镜',
          route: '/magnifier',
        ),
        MenuItemData(
          icon: Icons.tune_outlined,
          title: '滤镜调节',
          route: '/filter',
        ),
        MenuItemData(
          icon: Icons.bar_chart_outlined,
          title: '指标评估',
          route: '/metrics',
        ),
        MenuItemData(
          icon: Icons.info_outline,
          title: '算法信息',
          route: '/algorithm',
        ),
      ],
    ),
    MenuSection(
      title: '数据管理',
      items: [
        MenuItemData(
          icon: Icons.storage_outlined,
          title: '数据集管理',
          route: '/dataset',
        ),
      ],
    ),
  ];

  /// 获取所有菜单项（平铺）
  static List<MenuItemData> get allMenuItems {
    final items = <MenuItemData>[];
    for (final section in menuSections) {
      items.addAll(section.items);
    }
    return items;
  }

  /// 根据路由查找菜单项
  static MenuItemData? findMenuItemByRoute(String route) {
    for (final item in allMenuItems) {
      if (item.route == route) {
        return item;
      }
    }
    return null;
  }

  /// 检查路由是否存在于菜单中
  static bool containsRoute(String route) => findMenuItemByRoute(route) != null;
}