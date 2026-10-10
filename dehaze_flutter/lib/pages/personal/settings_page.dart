import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../widgets/logout_confirm_dialog.dart';
import '../../widgets/notification_settings_section.dart';

/// 系统设置 — L2 页面
///
/// 通知偏好（与消息设置页共用同一实现）+ 账号安全、缓存清理、关于、退出登录
class SettingsPage extends ConsumerStatefulWidget {
  const SettingsPage({super.key});

  @override
  ConsumerState<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends ConsumerState<SettingsPage> {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('系统设置')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // 账号安全
          _buildSection(theme, '账号', [
            _buildTile(Icons.person_outline, '个人信息', () {}),
            _buildTile(Icons.lock_outline, '修改密码', () {}),
          ]),
          const SizedBox(height: 16),
          // 通知偏好（与消息设置页共用）
          _buildSection(theme, '通知偏好', [const NotificationSettingsSection()]),
          const SizedBox(height: 16),
          // 缓存
          _buildSection(theme, '存储', [
            _buildTile(Icons.cleaning_services_outlined, '清除缓存', () {
              ScaffoldMessenger.of(
                context,
              ).showSnackBar(const SnackBar(content: Text('缓存已清除')));
            }),
          ]),
          const SizedBox(height: 16),
          // 关于
          _buildSection(theme, '其他', [
            _buildTile(
              Icons.info_outline,
              '关于我们',
              () => context.push(AppRouterConfig.about),
            ),
            _buildTile(Icons.description_outlined, '用户协议', () {}),
            _buildTile(Icons.privacy_tip_outlined, '隐私政策', () {}),
          ]),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => showLogoutConfirm(context, ref),
              icon: Icon(Icons.logout, color: theme.colorScheme.error),
              label: Text(
                '退出登录',
                style: TextStyle(color: theme.colorScheme.error),
              ),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                side: BorderSide(
                  color: theme.colorScheme.error.withValues(alpha: 0.3),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSection(ThemeData theme, String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(
            title,
            style: theme.textTheme.labelMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: theme.dividerColor),
          ),
          child: Column(children: children),
        ),
      ],
    );
  }

  Widget _buildTile(IconData icon, String title, VoidCallback onTap) {
    return ListTile(
      leading: Icon(icon, size: 22),
      title: Text(title),
      trailing: const Icon(Icons.chevron_right, size: 20),
      onTap: onTap,
      visualDensity: VisualDensity.compact,
    );
  }
}
