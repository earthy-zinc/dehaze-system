import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

class ToolsPage extends StatelessWidget {
  const ToolsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final isWide = ResponsiveUtils.isWideScreen(context);

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverToBoxAdapter(
            child: Padding(
              padding: EdgeInsets.fromLTRB(
                AppTheme.spacingM,
                AppTheme.spacingM,
                AppTheme.spacingM,
                0,
              ),
              child: _SearchBar(isWide: isWide),
            ),
          ),
          SliverToBoxAdapter(child: SizedBox(height: AppTheme.spacingL)),
          SliverToBoxAdapter(
            child: _QuickEntries(isWide: isWide),
          ),
          SliverToBoxAdapter(child: SizedBox(height: AppTheme.spacingL)),
          SliverToBoxAdapter(
            child: Padding(
              padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
              child: Text(
                '全部功能',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(child: SizedBox(height: AppTheme.spacingS)),
          SliverToBoxAdapter(
            child: _ToolGrid(isWide: isWide),
          ),
          SliverToBoxAdapter(
            child: SizedBox(
              height: AppTheme.spacingXXL + MediaQuery.of(context).padding.bottom,
            ),
          ),
        ],
      ),
    );
  }
}

class _SearchBar extends StatelessWidget {
  const _SearchBar({required this.isWide});

  final bool isWide;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      height: 48,
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.dividerColor),
      ),
      padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
      child: Row(
        children: [
          Icon(Icons.search, color: theme.colorScheme.onSurfaceVariant, size: 22),
          SizedBox(width: AppTheme.spacingS),
          Expanded(
            child: TextField(
              decoration: InputDecoration(
                hintText: '搜索算法、功能、文档...',
                hintStyle: TextStyle(
                  color: theme.colorScheme.onSurfaceVariant,
                  fontSize: 14,
                ),
                border: InputBorder.none,
                isDense: true,
                contentPadding: EdgeInsets.zero,
              ),
              style: theme.textTheme.bodyMedium,
            ),
          ),
        ],
      ),
    );
  }
}

class _QuickEntries extends StatelessWidget {
  const _QuickEntries({required this.isWide});

  final bool isWide;

  @override
  Widget build(BuildContext context) {
    final entries = [
      _QuickEntryData(
        icon: Icons.history,
        label: '处理历史',
        route: AppRouterConfig.taskHistory,
      ),
      _QuickEntryData(
        icon: Icons.favorite_border,
        label: '我的收藏',
        route: AppRouterConfig.profile,
      ),
      _QuickEntryData(
        icon: Icons.dashboard,
        label: '批量处理',
        route: AppRouterConfig.imageInput,
      ),
      _QuickEntryData(
        icon: Icons.psychology,
        label: '算法选择',
        route: AppRouterConfig.algorithmSelect,
      ),
      _QuickEntryData(
        icon: Icons.image,
        label: '图像输入',
        route: AppRouterConfig.imageInput,
      ),
    ];

    final theme = Theme.of(context);

    return SizedBox(
      height: 72,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
        itemCount: entries.length,
        separatorBuilder: (_, _) => SizedBox(width: 16),
        itemBuilder: (context, index) {
          final entry = entries[index];
          return GestureDetector(
            onTap: () => context.go(entry.route),
            child: SizedBox(
              width: 72,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: AppTheme.toolCardGradient,
                      ),
                      borderRadius: BorderRadius.circular(AppTheme.radiusM),
                    ),
                    child: Icon(
                      entry.icon,
                      color: AppTheme.brandBlue,
                      size: 22,
                    ),
                  ),
                  SizedBox(height: AppTheme.spacingXS),
                  Text(
                    entry.label,
                    style: theme.textTheme.labelSmall?.copyWith(
                      fontSize: 11,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

class _QuickEntryData {
  const _QuickEntryData({
    required this.icon,
    required this.label,
    required this.route,
  });

  final IconData icon;
  final String label;
  final String route;
}

class _ToolGrid extends StatelessWidget {
  const _ToolGrid({required this.isWide});

  final bool isWide;

  @override
  Widget build(BuildContext context) {
    final items = [
      _ToolItem(
        icon: Icons.image_outlined,
        title: '图像输入',
        subtitle: '上传、拍照、样例图片',
        route: AppRouterConfig.imageInput,
        color: AppTheme.brandBlue,
      ),
      _ToolItem(
        icon: Icons.psychology_outlined,
        title: '算法库',
        subtitle: '浏览和选择去雾算法',
        route: AppRouterConfig.algorithmSelect,
        color: AppTheme.techGreen,
      ),
      _ToolItem(
        icon: Icons.storage_outlined,
        title: '数据集',
        subtitle: '管理去雾数据集',
        route: AppRouterConfig.dataset,
        color: AppTheme.indigo,
      ),
      _ToolItem(
        icon: Icons.dashboard_outlined,
        title: '批量处理',
        subtitle: '批量上传和执行',
        route: AppRouterConfig.imageInput,
        color: AppTheme.warningColor,
      ),
      _ToolItem(
        icon: Icons.bar_chart_outlined,
        title: '指标管理',
        subtitle: 'PSNR/SSIM 评估',
        route: AppRouterConfig.compareMetrics,
        color: AppTheme.teal,
      ),
      _ToolItem(
        icon: Icons.description_outlined,
        title: 'API文档',
        subtitle: '开放接口文档',
        route: AppRouterConfig.home,
        color: AppTheme.successColor,
      ),
    ];

    final crossAxisCount = isWide ? 4 : 3;

    return Padding(
      padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
      child: GridView.builder(
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: crossAxisCount,
          crossAxisSpacing: AppTheme.spacingM,
          mainAxisSpacing: AppTheme.spacingM,
          // childAspectRatio 降低让 cell 更高，避免图标+title+subtitle 总高度溢出
          // （窄屏 3 列 cell 宽 ~80，原 1.05 → 高 76，可用高度仅 ~44，子内容需 ~132）
          childAspectRatio: isWide ? 1.0 : 0.82,
        ),
        itemCount: items.length,
        itemBuilder: (context, index) => _ToolCard(item: items[index]),
      ),
    );
  }
}

class _ToolItem {
  const _ToolItem({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.route,
    required this.color,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final String route;
  final Color color;
}

class _ToolCard extends StatelessWidget {
  const _ToolCard({required this.item});

  final _ToolItem item;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: InkWell(
        onTap: () => context.go(item.route),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        child: Padding(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: item.color.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                ),
                child: Icon(item.icon, color: item.color, size: 24),
              ),
              SizedBox(height: AppTheme.spacingS),
              Text(
                item.title,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
                textAlign: TextAlign.center,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              SizedBox(height: AppTheme.spacingXS),
              // Flexible 让 subtitle 在 cell 高度不足时自动省略，避免 Column 溢出
              Flexible(
                child: Text(
                  item.subtitle,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontSize: 11,
                  ),
                  textAlign: TextAlign.center,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
