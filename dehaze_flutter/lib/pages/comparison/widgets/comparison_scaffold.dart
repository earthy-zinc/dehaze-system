import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../router/config.dart';
import '../../../theme/app_theme.dart';

/// 效果对比页统一脚手架
///
/// 对齐产品文档 F-M05-007「对比工具栏」：
/// - 顶部：图标 + 标题 + 副标题
/// - 底部：模式切换行 + 快捷操作行（保存/分享/重新处理/更换算法/导出报告/收藏）
///
/// 各对比子页只需提供 [body] 与可选 [controls]，
/// 不再各自重复实现 header / bottomNav / noData。
class ComparisonScaffold extends StatelessWidget {
  const ComparisonScaffold({
    super.key,
    required this.icon,
    required this.title,
    this.subtitle,
    required this.body,
    this.controls,
    required this.currentRoute,
  });

  /// 标题图标
  final IconData icon;

  /// 标题文案
  final String title;

  /// 标题右侧副标题（操作提示等），为 null 时不渲染
  final String? subtitle;

  /// 主体内容
  final Widget body;

  /// 可选的参数控制区（如透明度滑块），渲染在 body 与底部工具栏之间
  final Widget? controls;

  /// 当前页路由，用于从模式切换中排除自身
  final String currentRoute;

  /// 全部对比模式（路由, 显示名）
  static final _allModes = <(String, String)>[
    (AppRouterConfig.sideBySide, '并排对比'),
    (AppRouterConfig.overlay, '重叠对比'),
    (AppRouterConfig.magnifier, '放大镜'),
    (AppRouterConfig.filter, '滤镜调节'),
    (AppRouterConfig.compareMetrics, '指标评估'),
    (AppRouterConfig.algorithm, '算法信息'),
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme),
          Expanded(child: body),
          if (controls != null) controls!,
          _buildBottomBar(context, theme),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(bottom: BorderSide(color: theme.dividerColor)),
        ),
        child: Row(
          children: [
            Icon(icon, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text(title,
                style: theme.textTheme.titleLarge
                    ?.copyWith(fontWeight: FontWeight.w700)),
            const Spacer(),
            if (subtitle != null)
              Text(subtitle!,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );

  Widget _buildBottomBar(BuildContext context, ThemeData theme) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(top: BorderSide(color: theme.dividerColor)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 模式切换行（排除当前页）
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 8,
              children: [
                for (final (route, label) in _allModes)
                  if (route != currentRoute)
                    ActionChip(
                      label: Text(label),
                      onPressed: () => context.go(route),
                    ),
              ],
            ),
            const SizedBox(height: 8),
            // 快捷操作行（对齐 F-M05-007）
            _buildQuickActions(context, theme),
          ],
        ),
      );

  /// 快捷操作工具栏
  ///
  /// 其中「重新处理」「更换算法」接通真实跳转；
  /// 保存/分享/导出/收藏当前无后端接口，先以 toast 提示「功能开发中」并保持可点击，
  /// 避免出现死入口（项目约束：导航参数必须完整传递，避免死入口或空白页面）。
  Widget _buildQuickActions(BuildContext context, ThemeData theme) {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 4,
      runSpacing: 4,
      children: [
        _QuickAction(
          icon: Icons.refresh,
          label: '重新处理',
          onTap: () => context.go(AppRouterConfig.processing),
        ),
        _QuickAction(
          icon: Icons.bolt,
          label: '更换算法',
          onTap: () => context.go(AppRouterConfig.algorithmSelect),
        ),
        _QuickAction(
          icon: Icons.save_outlined,
          label: '保存结果',
          onTap: () => _toast(context, '功能开发中'),
        ),
        _QuickAction(
          icon: Icons.ios_share_outlined,
          label: '分享图片',
          onTap: () => _toast(context, '功能开发中'),
        ),
        _QuickAction(
          icon: Icons.description_outlined,
          label: '导出报告',
          onTap: () => _toast(context, '功能开发中'),
        ),
        _QuickAction(
          icon: Icons.star_border,
          label: '收藏',
          onTap: () => _toast(context, '功能开发中'),
        ),
      ],
    );
  }

  void _toast(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), behavior: SnackBarBehavior.floating),
    );
  }
}

class _QuickAction extends StatelessWidget {
  const _QuickAction({
    required this.icon,
    required this.label,
    required this.onTap,
  });
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20),
            const SizedBox(height: 2),
            Text(label, style: const TextStyle(fontSize: 11)),
          ],
        ),
      ),
    );
  }
}
