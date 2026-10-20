import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../providers/auth_provider.dart';
import '../router/config.dart';

/// 显示登出确认对话框
///
/// 统一登出交互：弹出确认 → 调用 logout → 跳转首页。
/// 供 MainLayout 侧栏/抽屉 与 Profile 页共用，避免重复实现。
void showLogoutConfirm(BuildContext context, WidgetRef ref) {
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
