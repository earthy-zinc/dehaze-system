import 'package:flutter/material.dart';

/// 菜单项数据模型
class MenuItemData {
  const MenuItemData({
    required this.icon,
    required this.title,
    required this.route,
    this.badge,
    this.isNew = false,
  });

  final IconData icon;
  final String title;
  final String route;
  final String? badge; // 角标文字（如 "NEW"、数字等）
  final bool isNew; // 是否为新功能
}

/// 菜单分组数据模型
class MenuSection {
  const MenuSection({
    required this.title,
    required this.items,
    this.icon,
  });

  final String title;
  final List<MenuItemData> items;
  final IconData? icon; // 分组图标（可选）
}

/// 侧边菜单配置
///
/// 统一管理所有菜单项数据，便于维护和修改
/// 与 router/config.dart 路由配置保持一致
class MenuConfig {
  const MenuConfig._();

  /// 首页菜单项
  static const MenuItemData homeItem = MenuItemData(
    icon: Icons.home_outlined,
    title: '首页',
    route: '/home',
  );

  /// 获取所有菜单分组
  static const List<MenuSection> menuSections = [
    MenuSection(
      title: '处理流程',
      icon: Icons.play_circle_outline,
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
      icon: Icons.compare_outlined,
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
      icon: Icons.folder_outlined,
      items: [
        MenuItemData(
          icon: Icons.storage_outlined,
          title: '数据集管理',
          route: '/dataset',
        ),
      ],
    ),
  ];

  /// 获取所有菜单项（平铺，包含首页）
  static List<MenuItemData> get allMenuItems {
    final items = <MenuItemData>[homeItem];
    for (final section in menuSections) {
      items.addAll(section.items);
    }
    return items;
  }

  /// 获取所有菜单项（不包含首页）
  static List<MenuItemData> get menuItemsWithoutHome {
    final items = <MenuItemData>[];
    for (final section in menuSections) {
      items.addAll(section.items);
    }
    return items;
  }

  /// 根据路由查找菜单项
  static MenuItemData? findMenuItemByRoute(String route) {
    if (homeItem.route == route) return homeItem;
    for (final item in menuItemsWithoutHome) {
      if (item.route == route) {
        return item;
      }
    }
    return null;
  }

  /// 检查路由是否存在于菜单中
  static bool containsRoute(String route) => findMenuItemByRoute(route) != null;

  /// 根据路由获取所属分组
  static MenuSection? findSectionByRoute(String route) {
    for (final section in menuSections) {
      for (final item in section.items) {
        if (item.route == route) {
          return section;
        }
      }
    }
    return null;
  }

  /// 获取菜单项的索引（用于底部导航栏等）
  static int getMenuItemIndex(String route) {
    final items = allMenuItems;
    for (var i = 0; i < items.length; i++) {
      if (items[i].route == route) {
        return i;
      }
    }
    return 0;
  }
}