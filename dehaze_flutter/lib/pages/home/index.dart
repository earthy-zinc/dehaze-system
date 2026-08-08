import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import 'algorithm_section.dart';
import 'cta_section.dart';
import 'hero_section.dart';
import 'showcase_section.dart';
import 'tools_grid_section.dart';
import 'workflow_section.dart';

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isWide = ResponsiveUtils.isWideScreen(context);
    final sectionSpacing = AppTheme.spacingXL * 1.5;

    if (isWide) {
      return _buildDesktopLayout(context, sectionSpacing);
    }

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          const SliverToBoxAdapter(child: HeroSection()),
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),
          const SliverToBoxAdapter(child: ShowcaseSection()),
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),
          const SliverToBoxAdapter(child: WorkflowSection()),
          SliverToBoxAdapter(child: SizedBox(height: AppTheme.spacingXL)),
          const SliverToBoxAdapter(child: ToolsGridSection()),
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),
          const SliverToBoxAdapter(child: AlgorithmSection()),
          SliverToBoxAdapter(child: SizedBox(height: sectionSpacing)),
          const SliverToBoxAdapter(child: CTASection()),
          SliverToBoxAdapter(
            child: SizedBox(
              height: MediaQuery.of(context).padding.bottom + AppTheme.spacingXL,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDesktopLayout(BuildContext context, double sectionSpacing) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(AppTheme.spacingL),
      // crossAxisAlignment: stretch 让子 widget 在水平方向铺满父级宽度，
      // 否则 Column 默认收缩到子内容最小宽度，桌面端大窗口下 4 个统计卡片只能分到 ~85 像素
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildDesktopWelcome(context),
          SizedBox(height: AppTheme.spacingL),
          _buildDesktopStats(context),
          SizedBox(height: AppTheme.spacingL),
          _buildDesktopQuickActions(context),
          SizedBox(height: AppTheme.spacingL),
          const ToolsGridSection(),
          SizedBox(height: AppTheme.spacingXL),
        ],
      ),
    );
  }

  Widget _buildDesktopWelcome(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingL),
      decoration: BoxDecoration(
        gradient: AppTheme.getPrimaryGradient(),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        boxShadow: AppTheme.shadowLevel3,
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '欢迎使用图像去雾系统',
                  style: theme.textTheme.headlineSmall?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                SizedBox(height: AppTheme.spacingS),
                Text(
                  '采用先进的深度学习算法，一键还原清晰视界',
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: Colors.white.withValues(alpha: 0.85),
                  ),
                ),
              ],
            ),
          ),
          SizedBox(width: AppTheme.spacingL),
          FilledButton.icon(
            onPressed: () => context.go(AppRouterConfig.dehaze),
            icon: Icon(Icons.auto_fix_high, size: 20),
            label: Text('开始去雾'),
            style: FilledButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: AppTheme.brandBlue,
              padding: EdgeInsets.symmetric(
                horizontal: AppTheme.spacingL,
                vertical: AppTheme.spacingM,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDesktopStats(BuildContext context) {
    final theme = Theme.of(context);

    // TODO: 接入真实 API 获取统计数据
    // - 数据集总数: DatasetService.getPage() 聚合
    // - 可用算法: AlgorithmService.getPublicList() 聚合
    // - 任务总数/已完成: PredictionService.getPredictionLogs() 聚合
    // 当前后端暂无统一 Dashboard 聚合接口，暂时保留占位数据。
    final stats = [
      _StatItem(
        icon: Icons.storage_outlined,
        label: '数据集总数',
        value: '--',
        color: AppTheme.brandBlue,
      ),
      _StatItem(
        icon: Icons.psychology_outlined,
        label: '可用算法',
        value: '--',
        color: AppTheme.techGreen,
      ),
      _StatItem(
        icon: Icons.task_outlined,
        label: '任务总数',
        value: '--',
        color: AppTheme.indigo,
      ),
      _StatItem(
        icon: Icons.check_circle_outlined,
        label: '已完成',
        value: '--',
        color: AppTheme.successColor,
      ),
    ];

    return Row(
      children: stats.asMap().entries.map((entry) {
        final isLast = entry.key == stats.length - 1;
        return Expanded(
          child: Padding(
            // 最后一个 item 不加 right padding，避免多出一个间距导致溢出
            padding: EdgeInsets.only(right: isLast ? 0 : AppTheme.spacingM),
            child: _buildStatCard(context, theme, entry.value),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildStatCard(BuildContext context, ThemeData theme, _StatItem stat) {
    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingM),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: stat.color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(AppTheme.radiusM),
              ),
              child: Icon(stat.icon, color: stat.color, size: 24),
            ),
            SizedBox(width: AppTheme.spacingM),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    stat.value,
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    stat.label,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDesktopQuickActions(BuildContext context) {
    final theme = Theme.of(context);

    final actions = [
      _QuickAction(
        icon: Icons.image_outlined,
        label: '图像输入',
        color: AppTheme.brandBlue,
        route: AppRouterConfig.imageInput,
      ),
      _QuickAction(
        icon: Icons.psychology_outlined,
        label: '算法选择',
        color: AppTheme.techGreen,
        route: AppRouterConfig.algorithmSelect,
      ),
      _QuickAction(
        icon: Icons.storage_outlined,
        label: '数据集',
        color: AppTheme.indigo,
        route: AppRouterConfig.dataset,
      ),
      _QuickAction(
        icon: Icons.history,
        label: '处理历史',
        color: AppTheme.warningColor,
        route: AppRouterConfig.taskHistory,
      ),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '快捷操作',
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        SizedBox(height: AppTheme.spacingM),
        Row(
          children: actions.asMap().entries.map((entry) {
            final isLast = entry.key == actions.length - 1;
            return Expanded(
              child: Padding(
                // 最后一个 item 不加 right padding，避免多出一个间距
                padding: EdgeInsets.only(right: isLast ? 0 : AppTheme.spacingM),
                child: Card(
                  elevation: 1,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppTheme.radiusL),
                  ),
                  child: InkWell(
                    onTap: () => context.go(entry.value.route),
                    borderRadius: BorderRadius.circular(AppTheme.radiusL),
                    child: Padding(
                      padding: EdgeInsets.symmetric(
                        horizontal: AppTheme.spacingM,
                        vertical: AppTheme.spacingL,
                      ),
                      child: Column(
                        children: [
                          Icon(entry.value.icon, color: entry.value.color, size: 32),
                          SizedBox(height: AppTheme.spacingS),
                          Text(
                            entry.value.label,
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.w500,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }
}

class _StatItem {
  const _StatItem({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  final IconData icon;
  final String label;
  final String value;
  final Color color;
}

class _QuickAction {
  const _QuickAction({
    required this.icon,
    required this.label,
    required this.color,
    required this.route,
  });

  final IconData icon;
  final String label;
  final Color color;
  final String route;
}
