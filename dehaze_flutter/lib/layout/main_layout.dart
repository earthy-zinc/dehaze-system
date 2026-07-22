import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../constants/app_constants.dart';
import '../models/user_model.dart';
import '../providers/auth_provider.dart';
import '../router/config.dart';
import '../theme/app_theme.dart';
import 'menu_config.dart';

/// 主布局组件
///
/// 提供响应式布局，支持：
/// - 移动端：底部导航栏 + 抽屉菜单
/// - 平板/桌面：侧边导航栏
class MainLayout extends ConsumerWidget {
  const MainLayout({required this.child, super.key});

  final Widget child;

  /// 判断是否为宽屏设备（平板/桌面）
  bool _isWideScreen(BuildContext context) =>
      MediaQuery.of(context).size.width >= 768;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isWide = _isWideScreen(context);

    return Scaffold(
      appBar: _buildAppBar(context, ref),
      drawer: isWide ? null : _buildDrawer(context, ref),
      body: isWide ? _buildWideLayout(context, ref) : child,
      bottomNavigationBar: isWide ? null : _buildBottomNav(context),
    );
  }

  /// 构建顶部导航栏
  PreferredSizeWidget _buildAppBar(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authProvider);
    final user = authState.user;

    return AppBar(
      title: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              gradient: AppTheme.getPrimaryGradient(),
              borderRadius: BorderRadius.circular(8),
              boxShadow: AppTheme.shadowLevel1,
            ),
            child: const Icon(
              Icons.cloud_outlined,
              color: Colors.white,
              size: 18,
            ),
          ),
          const SizedBox(width: 8),
          const Text(AppConstants.appName),
        ],
      ),
      actions: [
        // 用户信息/登录按钮
        if (user != null)
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: GestureDetector(
              onTap: () => context.go(AppRouterConfig.profile),
              child: Center(
                child: Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: AppTheme.brandBlue,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Center(
                    child: Text(
                      user.avatarInitials,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          )
        else
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: TextButton.icon(
              onPressed: () => context.go(AppRouterConfig.login),
              icon: const Icon(Icons.login, size: 18),
              label: const Text('登录'),
            ),
          ),
      ],
    );
  }

  /// 构建宽屏布局（侧边栏 + 内容区）
  Widget _buildWideLayout(BuildContext context, WidgetRef ref) => Row(
        children: [
          // 侧边导航栏
          _buildSideNav(context, ref),
          // 内容区
          Expanded(child: child),
        ],
      );

  /// 构建侧边导航栏（平板/桌面）
  Widget _buildSideNav(BuildContext context, WidgetRef ref) {
    final currentLocation = GoRouterState.of(context).uri.toString();
    final authState = ref.watch(authProvider);
    final user = authState.user;

    return Container(
      width: 260,
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        border: Border(
          right: BorderSide(
            color: Theme.of(context).dividerColor,
            width: 1,
          ),
        ),
      ),
      child: Column(
        children: [
          // 导航菜单
          Expanded(
            child: ListView(
              padding: EdgeInsets.symmetric(vertical: AppTheme.spacingM),
              children: [
                // 首页
            _buildNavItem(
              context,
              MenuConfig.homeItem,
              isActive: currentLocation.startsWith(AppRouterConfig.home),
            ),
                const Divider(),
                // 分组菜单
                ...MenuConfig.menuSections.map(
                  (section) =>
                      _buildNavSection(context, section, currentLocation),
                ),
              ],
            ),
          ),
          // 底部用户信息
          _buildUserFooter(context, ref, user),
        ],
      ),
    );
  }

  /// 构建底部用户信息
  Widget _buildUserFooter(
    BuildContext context,
    WidgetRef ref,
    UserModel? user,
  ) {
    final theme = Theme.of(context);

    if (user == null) {
      return Container(
        padding: EdgeInsets.all(AppTheme.spacingM),
        decoration: BoxDecoration(
          border: Border(
            top: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            onPressed: () => context.go(AppRouterConfig.login),
            icon: const Icon(Icons.login, size: 18),
            label: const Text('登录'),
          ),
        ),
      );
    }

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: AppTheme.spacingM,
        vertical: AppTheme.spacingS,
      ),
      decoration: BoxDecoration(
        border: Border(
          top: BorderSide(color: theme.dividerColor, width: 1),
        ),
      ),
      child: Row(
        children: [
          // 头像
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: AppTheme.brandBlue,
              borderRadius: BorderRadius.circular(18),
            ),
            child: Center(
              child: Text(
                user.avatarInitials,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          // 用户名
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  user.nickname ?? user.username,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                if (user.roles.isNotEmpty)
                  Text(
                    user.roles.first,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
              ],
            ),
          ),
          // 登出按钮
          IconButton(
            icon: const Icon(Icons.logout, size: 20),
            tooltip: '退出登录',
            onPressed: () => _showLogoutConfirm(context, ref),
          ),
        ],
      ),
    );
  }

  /// 显示登出确认对话框
  void _showLogoutConfirm(BuildContext context, WidgetRef ref) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('退出登录'),
        content: const Text('确定要退出登录吗？'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.of(ctx).pop();
              await ref.read(authProvider.notifier).logout();
              if (context.mounted) {
                context.go(AppRouterConfig.home);
              }
            },
            child: const Text('确定'),
          ),
        ],
      ),
    );
  }

  /// 构建导航分组
  Widget _buildNavSection(
    BuildContext context,
    MenuSection section,
    String currentLocation,
  ) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Padding(
        padding: EdgeInsets.symmetric(
          horizontal: AppTheme.spacingM,
          vertical: AppTheme.spacingS,
        ),
        child: Row(
          children: [
            if (section.icon != null) ...[
              Icon(
                section.icon,
                size: 16,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              const SizedBox(width: 8),
            ],
            Text(
              section.title,
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
                fontWeight: FontWeight.w600,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ),
      ),
      ...section.items.map(
        (item) => _buildNavItem(
          context,
          item,
          isActive: currentLocation.startsWith(item.route),
        ),
      ),
      SizedBox(height: AppTheme.spacingS),
    ],
  );

  /// 构建单个导航项
  Widget _buildNavItem(
    BuildContext context,
    MenuItemData item, {
    bool isActive = false,
  }) {
    final colorScheme = Theme.of(context).colorScheme;

    return Padding(
      padding: EdgeInsets.symmetric(
        horizontal: AppTheme.spacingS,
        vertical: 2,
      ),
      child: Material(
        color: isActive
            ? colorScheme.primary.withValues(alpha: 0.1)
            : Colors.transparent,
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: InkWell(
          onTap: () => context.go(item.route),
          borderRadius: BorderRadius.circular(AppTheme.radiusM),
          child: Padding(
            padding: EdgeInsets.symmetric(
              horizontal: AppTheme.spacingM,
              vertical: AppTheme.spacingS,
            ),
            child: Row(
              children: [
                Icon(
                  item.icon,
                  size: 20,
                  color: isActive
                      ? colorScheme.primary
                      : colorScheme.onSurfaceVariant,
                ),
                SizedBox(width: AppTheme.spacingM),
                Expanded(
                  child: Text(
                    item.title,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: isActive
                          ? colorScheme.primary
                          : colorScheme.onSurface,
                      fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                ),
                if (item.badge != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 6,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: colorScheme.primary,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      item.badge!,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  /// 构建抽屉菜单（移动端）
  Widget _buildDrawer(BuildContext context, WidgetRef ref) {
    final currentLocation = GoRouterState.of(context).uri.toString();
    final authState = ref.watch(authProvider);
    final user = authState.user;

    return Drawer(
      child: SafeArea(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            // 抽屉头部
            Container(
              padding: EdgeInsets.all(AppTheme.spacingL),
              decoration: BoxDecoration(
                gradient: AppTheme.getPrimaryGradient(),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 48,
                        height: 48,
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: user != null
                            ? Center(
                                child: Text(
                                  user.avatarInitials,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 20,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              )
                            : const Icon(
                                Icons.cloud_outlined,
                                color: Colors.white,
                                size: 28,
                              ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              user?.nickname ??
                                  user?.username ??
                                  AppConstants.appName,
                              style:
                                  Theme.of(context).textTheme.titleLarge?.copyWith(
                                color: Colors.white,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            if (user != null && user.roles.isNotEmpty)
                              Text(
                                user.roles.first,
                                style:
                                    Theme.of(context).textTheme.bodySmall?.copyWith(
                                  color: Colors.white.withValues(alpha: 0.8),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    user != null ? '功能菜单' : '点击登录以使用完整功能',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.white.withValues(alpha: 0.8),
                    ),
                  ),
                  if (user == null) ...[
                    const SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton.icon(
                        onPressed: () {
                          Navigator.pop(context);
                          context.go(AppRouterConfig.login);
                        },
                        icon: const Icon(Icons.login, size: 18),
                        label: const Text('登录'),
                        style: FilledButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: AppTheme.brandBlue,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
            // 首页
            _buildDrawerItem(
              context,
              MenuConfig.homeItem,
              isActive: currentLocation.startsWith(AppRouterConfig.home),
            ),
            const Divider(),
            // 分组菜单
            ...MenuConfig.menuSections.expand(
              (section) => [
                Padding(
                  padding: EdgeInsets.fromLTRB(
                    AppTheme.spacingM,
                    AppTheme.spacingM,
                    AppTheme.spacingM,
                    AppTheme.spacingS,
                  ),
                  child: Text(
                    section.title,
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
                ...section.items.map(
                  (item) => _buildDrawerItem(
                    context,
                    item,
                    isActive: currentLocation.startsWith(item.route),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// 构建抽屉菜单项
  Widget _buildDrawerItem(
    BuildContext context,
    MenuItemData item, {
    bool isActive = false,
  }) {
    final colorScheme = Theme.of(context).colorScheme;

    return ListTile(
      leading: Icon(
        item.icon,
        color: isActive ? colorScheme.primary : colorScheme.onSurfaceVariant,
      ),
      title: Text(
        item.title,
        style: TextStyle(
          color: isActive ? colorScheme.primary : colorScheme.onSurface,
          fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
        ),
      ),
      selected: isActive,
      selectedTileColor: colorScheme.primary.withValues(alpha: 0.1),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
      ),
      onTap: () {
        Navigator.pop(context); // 关闭 Drawer
        context.go(item.route); // 跳转
      },
    );
  }

  /// 构建底部导航栏（移动端）
  Widget _buildBottomNav(BuildContext context) {
    final currentIndex = _getCurrentIndex(context);

    return NavigationBar(
      selectedIndex: currentIndex,
      onDestinationSelected: (index) {
        switch (index) {
          case 0:
            context.go(AppRouterConfig.home);
            break;
          case 1:
            context.go(AppRouterConfig.dataset);
            break;
          case 2:
            context.go(AppRouterConfig.profile);
            break;
          case 3:
            context.go(AppRouterConfig.taskHistory);
            break;
        }
      },
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.home_outlined),
          selectedIcon: Icon(Icons.home),
          label: '首页',
        ),
        NavigationDestination(
          icon: Icon(Icons.storage_outlined),
          selectedIcon: Icon(Icons.storage),
          label: '数据集',
        ),
        NavigationDestination(
          icon: Icon(Icons.person_outline),
          selectedIcon: Icon(Icons.person),
          label: '我的',
        ),
        NavigationDestination(
          icon: Icon(Icons.history_outlined),
          selectedIcon: Icon(Icons.history),
          label: '历史',
        ),
      ],
    );
  }

  int _getCurrentIndex(BuildContext context) {
    final location = GoRouterState.of(context).uri.toString();
    if (location.startsWith(AppRouterConfig.home)) return 0;
    if (location.startsWith(AppRouterConfig.dataset)) return 1;
    if (location.startsWith(AppRouterConfig.profile)) return 2;
    if (location.startsWith(AppRouterConfig.taskHistory)) return 3;
    return 0;
  }
}
