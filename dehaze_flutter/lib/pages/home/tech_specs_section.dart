import 'package:flutter/material.dart';

import '../../theme/app_theme.dart';

/// 技术特性区域组件
///
/// 展示应用的技术规格和性能指标
class TechSpecsSection extends StatelessWidget {
  const TechSpecsSection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final specs = [
      {
        'icon': Icons.bolt,
        'title': '高性能',
        'value': '60fps',
        'desc': '流畅运行，响应时间<200ms',
      },
      {
        'icon': Icons.smartphone,
        'title': '全平台',
        'value': '100%',
        'desc': '完美适配手机、平板、桌面',
      },
      {
        'icon': Icons.psychology,
        'title': '智能算法',
        'value': '8+',
        'desc': '支持多种先进去雾算法',
      },
      {
        'icon': Icons.bar_chart,
        'title': '专业评估',
        'value': '5+',
        'desc': '多维度定量分析指标',
      },
    ];

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingXL),
      color: theme.colorScheme.surface,
      child: LayoutBuilder(
        builder: (context, constraints) {
          // 根据屏幕宽度调整列数
          var crossAxisCount = 2;
          if (constraints.maxWidth > 600) {
            crossAxisCount = 4;
          }

          // 根据屏幕宽度调整宽高比，确保内容不溢出
          final childAspectRatio = crossAxisCount == 4 ? 0.85 : 0.9;

          return GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: crossAxisCount,
              crossAxisSpacing: AppTheme.spacingM,
              mainAxisSpacing: AppTheme.spacingM,
              childAspectRatio: childAspectRatio,
            ),
            itemCount: specs.length,
            itemBuilder: (context, index) {
              final spec = specs[index];
              return _SpecCard(
                icon: spec['icon']! as IconData,
                title: spec['title']! as String,
                value: spec['value']! as String,
                description: spec['desc']! as String,
              );
            },
          );
        },
      ),
    );
  }
}

/// 单个规格卡片组件
class _SpecCard extends StatelessWidget {
  const _SpecCard({
    required this.icon,
    required this.title,
    required this.value,
    required this.description,
  });

  final IconData icon;
  final String title;
  final String value;
  final String description;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.spacingL),
        side: BorderSide(color: theme.dividerColor, width: 1),
      ),
      child: Padding(
        padding: EdgeInsets.all(AppTheme.spacingM),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          mainAxisSize: MainAxisSize.min,
          children: [
            // 图标 - 使用 Flexible 防止溢出
            Flexible(
              flex: 3,
              child: Container(
                constraints: const BoxConstraints(
                  maxWidth: 56,
                  maxHeight: 56,
                ),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF3b82f6), Color(0xFF2563eb)],
                  ),
                  borderRadius: BorderRadius.circular(28),
                ),
                child: Center(
                  child: Icon(icon, color: Colors.white, size: 24),
                ),
              ),
            ),
            SizedBox(height: AppTheme.spacingS),

            // 标题
            Flexible(
              flex: 1,
              child: Text(
                title,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: theme.textTheme.bodyMedium?.color,
                ),
                textAlign: TextAlign.center,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            SizedBox(height: AppTheme.spacingXS),

            // 数值 - 使用渐变色
            Flexible(
              flex: 2,
              child: ShaderMask(
                shaderCallback: (bounds) => const LinearGradient(
                  colors: [Color(0xFF3b82f6), Color(0xFF2563eb)],
                ).createShader(bounds),
                child: Text(
                  value,
                  style: theme.textTheme.headlineSmall?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            SizedBox(height: AppTheme.spacingXS),

            // 描述
            Flexible(
              flex: 2,
              child: Text(
                description,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontSize: 11,
                  height: 1.3,
                  color: theme.textTheme.bodySmall?.color?.withValues(alpha: 0.7),
                ),
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
