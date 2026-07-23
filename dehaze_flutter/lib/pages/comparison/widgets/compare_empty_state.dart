import 'package:flutter/material.dart';

/// 对比页统一空状态组件
///
/// 用于对比页在缺少前置数据（如未完成去雾处理、无可用算法）时展示占位 UI，
/// 替代各对比页内重复的 `_buildNoData` 实现。
class CompareEmptyState extends StatelessWidget {
  const CompareEmptyState({
    super.key,
    this.icon = Icons.warning_amber,
    this.iconColor,
    this.message = '请先完成去雾处理',
    this.actionLabel = '去处理',
    required this.onAction,
  });

  /// 空状态图标
  final IconData icon;

  /// 图标颜色，默认取主题 error 色
  final Color? iconColor;

  /// 提示文案
  final String message;

  /// 操作按钮文案
  final String actionLabel;

  /// 操作按钮回调
  final VoidCallback onAction;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 64, color: iconColor ?? theme.colorScheme.error),
          const SizedBox(height: 16),
          Text(message, style: theme.textTheme.titleMedium),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: onAction,
            child: Text(actionLabel),
          ),
        ],
      ),
    );
  }
}
