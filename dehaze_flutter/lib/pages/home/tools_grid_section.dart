import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../theme/app_theme.dart';
import '../../router/config.dart';

/// 工具网格区域组件
///
/// 展示各种对比和分析工具
class ToolsGridSection extends StatelessWidget {
  const ToolsGridSection({super.key});

  @override
  Widget build(BuildContext context) {
    final tools = [
      {
        'icon': Icons.view_column,
        'title': '并排对比',
        'desc': '多图并排展示，支持2-4张图片同屏对比',
        'route': AppRouterConfig.sideBySide,
      },
      {
        'icon': Icons.layers,
        'title': '重叠对比',
        'desc': '拖动分割线实时对比，支持横向和纵向模式',
        'route': AppRouterConfig.overlay,
      },
      {
        'icon': Icons.search,
        'title': '放大镜',
        'desc': '局部细节放大查看，精确对比图像质量',
        'route': AppRouterConfig.magnifier,
      },
      {
        'icon': Icons.tune,
        'title': '滤镜调节',
        'desc': '实时调节亮度、对比度、饱和度等参数',
        'route': AppRouterConfig.filter,
      },
      {
        'icon': Icons.analytics,
        'title': '指标评估',
        'desc': 'SSIM、PSNR等专业指标定量分析',
        'route': AppRouterConfig.metrics,
      },
      {
        'icon': Icons.storage,
        'title': '数据集管理',
        'desc': '浏览和管理多个专业去雾数据集',
        'route': AppRouterConfig.dataset,
      },
    ];

    return Column(
      children: [
        // 标题区域已在WorkflowSection中处理，这里直接显示网格

        // 工具网格
        LayoutBuilder(
          builder: (context, constraints) {
            // 根据屏幕宽度调整列数
            int crossAxisCount = 2;
            if (constraints.maxWidth > 800) {
              crossAxisCount = 3;
            } else if (constraints.maxWidth > 1200) {
              crossAxisCount = 4;
            }

            return GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: crossAxisCount,
                crossAxisSpacing: AppTheme.spacingL,
                mainAxisSpacing: AppTheme.spacingL,
                childAspectRatio: 1.2,
              ),
              itemCount: tools.length,
              itemBuilder: (context, index) {
                final tool = tools[index];
                return _ToolCard(
                  icon: tool['icon']! as IconData,
                  title: tool['title']! as String,
                  description: tool['desc']! as String,
                  route: tool['route']! as String,
                );
              },
            );
          },
        ),
      ],
    );
  }
}

/// 单个工具卡片组件
class _ToolCard extends StatelessWidget {
  const _ToolCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.route,
  });

  final IconData icon;
  final String title;
  final String description;
  final String route;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.spacingXL),
        side: BorderSide(color: theme.dividerColor, width: 2),
      ),
      child: InkWell(
        onTap: () => context.go(route),
        borderRadius: BorderRadius.circular(AppTheme.spacingXL),
        child: Padding(
          padding: const EdgeInsets.all(AppTheme.spacingL),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // 图标容器
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFeff6ff), Color(0xFFdbeafe)],
                  ),
                  borderRadius: BorderRadius.circular(AppTheme.radiusL),
                ),
                child: Icon(icon, color: const Color(0xFF3b82f6), size: 24),
              ),
              const SizedBox(height: AppTheme.spacingL),

              // 标题
              Text(
                title,
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                ),
                textAlign: TextAlign.center,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: AppTheme.spacingM),

              // 描述
              Expanded(
                child: Text(
                  description,
                  style: theme.textTheme.bodySmall?.copyWith(height: 1.4),
                  textAlign: TextAlign.center,
                  maxLines: 3,
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
