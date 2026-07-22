import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/user_model.dart';
import '../../providers/auth_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 用户中心页面
class ProfilePage extends ConsumerWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authProvider);
    final user = authState.user;
    final theme = Theme.of(context);

    if (user == null) {
      return _buildNotLoggedIn(context, theme);
    }

    return Scaffold(
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // 用户信息头部
            _buildUserHeader(theme, user),
            const SizedBox(height: 16),
            // 角色标签
            _buildRolesSection(theme, user.roles),
            const SizedBox(height: 16),
            // 权限概览
            _buildPermissionsSection(theme, user.permissions),
            const SizedBox(height: 16),
            // 功能入口
            _buildMenuSection(theme, context),
            const SizedBox(height: 16),
            // 退出登录按钮
            _buildLogoutButton(context, ref, theme),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildUserHeader(ThemeData theme, UserModel user) => Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: AppTheme.getPrimaryGradient(),
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
        ),
        child: Row(
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(32),
              ),
              child: Center(
                child: Text(
                  user.avatarInitials,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    user.nickname ?? user.username,
                    style: theme.textTheme.titleLarge?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '@${user.username}',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: Colors.white.withValues(alpha: 0.8),
                    ),
                  ),
                  if (user.deptName != null) ...[
                    const SizedBox(height: 4),
                    Text(
                      user.deptName!,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: Colors.white.withValues(alpha: 0.7),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      );

  Widget _buildRolesSection(ThemeData theme, List<String> roles) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('角色', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: roles.map((role) {
                  final isRoot = role == 'ROOT';
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: (isRoot ? AppTheme.errorColor : AppTheme.brandBlue).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: (isRoot ? AppTheme.errorColor : AppTheme.brandBlue).withValues(alpha: 0.3),
                      ),
                    ),
                    child: Text(
                      role,
                      style: TextStyle(
                        color: isRoot ? AppTheme.errorColor : AppTheme.brandBlue,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  );
                }).toList(),
              ),
            ],
          ),
        ),
      );

  Widget _buildPermissionsSection(ThemeData theme, List<String> permissions) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('权限概览', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 4),
              Text(
                '共 ${permissions.length} 个权限',
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
              ),
              const SizedBox(height: 12),
              if (permissions.isEmpty)
                Text('暂无权限', style: theme.textTheme.bodyMedium)
              else
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: permissions.take(20).map((perm) {
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: theme.colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        perm,
                        style: TextStyle(fontSize: 11, color: theme.colorScheme.onSurfaceVariant),
                      ),
                    );
                  }).toList(),
                ),
              if (permissions.length > 20)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    '...还有 ${permissions.length - 20} 个权限',
                    style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                  ),
                ),
            ],
          ),
        ),
      );

  Widget _buildMenuSection(ThemeData theme, BuildContext context) => Card(
        child: Column(
          children: [
            ListTile(
              leading: const Icon(Icons.history),
              title: const Text('处理历史'),
              trailing: const Icon(Icons.chevron_right),
              onTap: () => context.go(AppRouterConfig.taskHistory),
            ),
          ],
        ),
      );

  Widget _buildLogoutButton(BuildContext context, WidgetRef ref, ThemeData theme) => SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          onPressed: () => _showLogoutConfirm(context, ref),
          icon: Icon(Icons.logout, color: theme.colorScheme.error),
          label: Text('退出登录', style: TextStyle(color: theme.colorScheme.error)),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 14),
            side: BorderSide(color: theme.colorScheme.error.withValues(alpha: 0.3)),
          ),
        ),
      );

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

  Widget _buildNotLoggedIn(BuildContext context, ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.person_off_outlined, size: 64, color: theme.colorScheme.onSurfaceVariant),
            const SizedBox(height: 16),
            Text('请先登录', style: theme.textTheme.titleMedium),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => context.go(AppRouterConfig.login),
              child: const Text('去登录'),
            ),
          ],
        ),
      );
}
