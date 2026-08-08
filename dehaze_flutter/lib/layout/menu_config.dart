import 'package:flutter/material.dart';

/// 菜单项数据模型
class MenuItemData {
  const MenuItemData({
    required this.icon,
    required this.selectedIcon,
    required this.title,
    required this.route,
    this.badge,
    this.children = const [],
  });

  final IconData icon;
  final IconData selectedIcon;
  final String title;
  final String route;
  final int? badge;
  final List<MenuItemData> children;
}

/// 桌面端侧边栏分组
class MenuGroup {
  const MenuGroup({required this.label, required this.items});
  final String label;
  final List<MenuItemData> items;
}

/// 菜单配置
///
/// 5 Tab 导航：首页 / 工具 / 去雾 / 消息 / 我的
/// 与 router/config.dart 路由配置保持一致
class MenuConfig {
  const MenuConfig._();

  // ==================== 5 Tab 定义 ====================

  static const List<MenuItemData> tabs = [
    MenuItemData(
      icon: Icons.home_outlined,
      selectedIcon: Icons.home,
      title: '首页',
      route: '/home',
    ),
    MenuItemData(
      icon: Icons.grid_view_outlined,
      selectedIcon: Icons.grid_view,
      title: '工具',
      route: '/tools',
    ),
    MenuItemData(
      icon: Icons.auto_fix_high_outlined,
      selectedIcon: Icons.auto_fix_high,
      title: '去雾',
      route: '/dehaze',
    ),
    MenuItemData(
      icon: Icons.notifications_outlined,
      selectedIcon: Icons.notifications,
      title: '消息',
      route: '/messages',
    ),
    MenuItemData(
      icon: Icons.person_outline,
      selectedIcon: Icons.person,
      title: '我的',
      route: '/profile',
    ),
  ];

  /// 根据路由获取当前 Tab 索引
  static int getTabIndex(String location) {
    for (int i = 0; i < tabs.length; i++) {
      if (location.startsWith(tabs[i].route)) return i;
    }
    return 0;
  }

  // ==================== 桌面端侧边栏分组 ====================

  static const List<MenuGroup> desktopGroups = [
    MenuGroup(label: '主功能', items: tabs),
  ];
}
