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
      padding: const EdgeInsets.all(AppTheme.spacingXL),
      color: theme.colorScheme.surface,
      child: LayoutBuilder(
        builder: (context, constraints) {
          // 根据屏幕宽度调整列数
          int crossAxisCount = 2;
          if (constraints.maxWidth > 600) {
            crossAxisCount = 4;
          }

          return GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: crossAxisCount,
              crossAxisSpacing: AppTheme.spacingL,
              mainAxisSpacing: AppTheme.spacingL,
              childAspectRatio: 1.1,
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
        borderRadius: BorderRadius.circular(AppTheme.spacingXL),
        side: BorderSide(color: theme.dividerColor, width: 2),
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppTheme.spacingL),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 图标
            Container(
              width: 72,
              height: 72,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF3b82f6), Color(0xFF2563eb)],
                ),
                borderRadius: BorderRadius.circular(36),
              ),
              child: Icon(icon, color: Colors.white, size: 32),
            ),
            const SizedBox(height: AppTheme.spacingL),

            // 标题
            Text(
              title,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
                color: theme.textTheme.bodyMedium?.color,
                letterSpacing: 0.05,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: AppTheme.spacingM),

            // 数值 - 使用渐变色
            ShaderMask(
              shaderCallback: (bounds) => const LinearGradient(
                colors: [Color(0xFF3b82f6), Color(0xFF2563eb)],
              ).createShader(bounds),
              child: Text(
                value,
                style: theme.textTheme.displayMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(height: AppTheme.spacingS),

            // 描述
            Text(
              description,
              style: theme.textTheme.bodySmall?.copyWith(height: 1.5),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
