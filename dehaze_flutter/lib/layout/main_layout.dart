import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../models/user_model.dart';
import '../providers/auth_provider.dart';
import '../router/config.dart';
import '../theme/app_theme.dart';
import '../widgets/logout_confirm_dialog.dart';
import 'menu_config.dart';

/// 主布局组件（响应式双布局）
///
/// 接收 [StatefulNavigationShell]，响应式切换布局：
/// - 移动端（< 768px）：Scaffold + NavigationBar（Material 3，5 Tab）
/// - 桌面端（≥ 768px）：Row（248px 侧边栏 + 内容区 + 顶栏面包屑）
class MainLayout extends ConsumerWidget {
  const MainLayout({required this.navigationShell, super.key});

  final StatefulNavigationShell navigationShell;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isWide = MediaQuery.sizeOf(context).width >= 768;
    return isWide ? _DesktopLayout(shell: navigationShell) : _MobileLayout(shell: navigationShell);
  }
}

// ==================== 移动端布局 ====================

class _MobileLayout extends ConsumerWidget {
  const _MobileLayout({required this.shell});

  final StatefulNavigationShell shell;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final currentIndex = shell.currentIndex;
    final isHome = currentIndex == 0;

    return Scaffold(
      appBar: AppBar(
        title: isHome
            ? Row(
                children: [
                  Container(
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      gradient: AppTheme.getPrimaryGradient(),
                      borderRadius: BorderRadius.circular(8),
                      boxShadow: AppTheme.shadowLevel1,
                    ),
                    child: const Icon(Icons.cloud_outlined, color: Colors.white, size: 18),
                  ),
                  const SizedBox(width: 8),
                  const Text('图像去雾系统'),
                ],
              )
            : Text(MenuConfig.tabs[currentIndex].title),
        centerTitle: false,
      ),
      body: shell,
      bottomNavigationBar: NavigationBar(
        selectedIndex: currentIndex,
        onDestinationSelected: (index) => _goBranch(index),
        destinations: [
          for (final tab in MenuConfig.tabs)
            NavigationDestination(
              icon: Icon(tab.icon),
              selectedIcon: Icon(tab.selectedIcon),
              label: tab.title,
            ),
        ],
      ),
    );
  }

  void _goBranch(int index) {
    shell.goBranch(
      index,
      initialLocation: index == shell.currentIndex,
    );
  }
}

// ==================== 桌面端布局 ====================

class _DesktopLayout extends ConsumerStatefulWidget {
  const _DesktopLayout({required this.shell});

  final StatefulNavigationShell shell;

  @override
  ConsumerState<_DesktopLayout> createState() => _DesktopLayoutState();
}

class _DesktopLayoutState extends ConsumerState<_DesktopLayout> {
  bool _sidebarCollapsed = false;

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final user = authState.user;
    final shell = widget.shell;
    final theme = Theme.of(context);
    final sidebarWidth = _sidebarCollapsed ? 64.0 : 248.0;

    return Scaffold(
      body: Row(
        children: [
          // 侧边栏
          _buildSidebar(theme, user, sidebarWidth),
          // 内容区
          Expanded(
            child: Column(
              children: [
                _buildTopBar(theme, user),
                Expanded(child: shell),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidebar(ThemeData theme, UserModel? user, double width) {
    final currentIndex = widget.shell.currentIndex;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: width,
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        border: Border(
          right: BorderSide(color: theme.dividerColor, width: 1),
        ),
      ),
      child: Column(
        children: [
          // Logo 区
          Container(
            height: 64,
            padding: EdgeInsets.symmetric(horizontal: _sidebarCollapsed ? 0 : 16),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(color: theme.dividerColor)),
            ),
            child: _sidebarCollapsed
                ? Center(
                    child: Container(
                      width: 36,
                      height: 36,
                      decoration: BoxDecoration(
                        gradient: AppTheme.getPrimaryGradient(),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Icon(Icons.cloud_outlined, color: Colors.white, size: 20),
                    ),
                  )
                : Row(
                    children: [
                      Container(
                        width: 36,
                        height: 36,
                        decoration: BoxDecoration(
                          gradient: AppTheme.getPrimaryGradient(),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Icon(Icons.cloud_outlined, color: Colors.white, size: 20),
                      ),
                      const SizedBox(width: 10),
                      const Expanded(
                        child: Text(
                          '图像去雾',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ),
          ),
          // 导航项
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(vertical: 8),
              children: [
                for (int i = 0; i < MenuConfig.tabs.length; i++)
                  _buildSidebarItem(theme, MenuConfig.tabs[i], isActive: i == currentIndex),
              ],
            ),
          ),
          // 底部用户信息
          _buildSidebarFooter(theme, user),
          // 折叠按钮
          InkWell(
            onTap: () => setState(() => _sidebarCollapsed = !_sidebarCollapsed),
            child: Container(
              height: 40,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                border: Border(top: BorderSide(color: theme.dividerColor)),
              ),
              child: Icon(
                _sidebarCollapsed ? Icons.chevron_right : Icons.chevron_left,
                size: 20,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSidebarItem(ThemeData theme, MenuItemData item, {bool isActive = false}) {
    final colorScheme = theme.colorScheme;

    return Padding(
      padding: EdgeInsets.symmetric(
        horizontal: _sidebarCollapsed ? 8.0 : 12.0,
        vertical: 2,
      ),
      child: Material(
        color: isActive ? colorScheme.primary.withValues(alpha: 0.1) : Colors.transparent,
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: InkWell(
          onTap: () {
            final index = MenuConfig.tabs.indexOf(item);
            widget.shell.goBranch(
              index,
              initialLocation: index == widget.shell.currentIndex,
            );
          },
          borderRadius: BorderRadius.circular(AppTheme.radiusM),
          child: _sidebarCollapsed
              ? SizedBox(
                  height: 48,
                  child: Center(
                    child: Icon(
                      isActive ? item.selectedIcon : item.icon,
                      size: 24,
                      color: isActive ? colorScheme.primary : colorScheme.onSurfaceVariant,
                    ),
                  ),
                )
              : Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  child: Row(
                    children: [
                      Icon(
                        isActive ? item.selectedIcon : item.icon,
                        size: 20,
                        color: isActive ? colorScheme.primary : colorScheme.onSurfaceVariant,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          item.title,
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: isActive ? colorScheme.primary : colorScheme.onSurface,
                            fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
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

  Widget _buildSidebarFooter(ThemeData theme, UserModel? user) {
    if (user == null) {
      return Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          border: Border(top: BorderSide(color: theme.dividerColor)),
        ),
        child: _sidebarCollapsed
            ? IconButton(
                icon: const Icon(Icons.login, size: 20),
                onPressed: () => _goLogin(),
              )
            : SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: () => _goLogin(),
                  icon: const Icon(Icons.login, size: 18),
                  label: const Text('登录'),
                ),
              ),
      );
    }

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: _sidebarCollapsed ? 0 : 12,
        vertical: 10,
      ),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: theme.dividerColor)),
      ),
      child: _sidebarCollapsed
          ? Center(
              child: GestureDetector(
                onTap: () => widget.shell.goBranch(4),
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
                      style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
              ),
            )
          : Row(
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: AppTheme.brandBlue,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Center(
                    child: Text(
                      user.avatarInitials,
                      style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        user.nickname ?? user.username,
                        style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (user.roleNames.isNotEmpty)
                        Text(
                          user.roleNames.first,
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.logout, size: 18),
                  tooltip: '退出登录',
                  onPressed: () => showLogoutConfirm(context, ref),
                ),
              ],
            ),
    );
  }

  Widget _buildTopBar(ThemeData theme, UserModel? user) {
    final currentIndex = widget.shell.currentIndex;
    final currentTab = MenuConfig.tabs[currentIndex];

    return Container(
      height: 56,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        border: Border(bottom: BorderSide(color: theme.dividerColor)),
      ),
      child: Row(
        children: [
          // 面包屑
          Text(
            currentTab.title,
            style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
          ),
          const Spacer(),
          // 通知按钮
          IconButton(
            icon: const Icon(Icons.notifications_outlined, size: 22),
            tooltip: '通知',
            onPressed: () => widget.shell.goBranch(3),
          ),
          const SizedBox(width: 4),
          // 用户头像
          if (user != null)
            GestureDetector(
              onTap: () => widget.shell.goBranch(4),
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
                    style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
                  ),
                ),
              ),
            )
          else
            TextButton.icon(
              onPressed: () => _goLogin(),
              icon: const Icon(Icons.login, size: 18),
              label: const Text('登录'),
            ),
        ],
      ),
    );
  }

  void _goLogin() {
    final router = GoRouter.of(context);
    router.go(AppRouterConfig.login);
  }
}
