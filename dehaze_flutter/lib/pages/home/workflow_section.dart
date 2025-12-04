import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 工作流程区域组件
///
/// 展示图像处理的三个主要步骤
class WorkflowSection extends StatelessWidget {
  const WorkflowSection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final steps = [
      {
        'number': '01',
        'icon': Icons.image,
        'title': '图像输入',
        'description': '支持上传、拍照、样例图片\n多种输入方式随心选择',
        'route': AppRouterConfig.imageInput,
      },
      {
        'number': '02',
        'icon': Icons.psychology,
        'title': '智能算法',
        'description': '多种去雾算法可选\nAI智能推荐最优方案',
        'route': AppRouterConfig.algorithmSelect,
      },
      {
        'number': '03',
        'icon': Icons.auto_fix_high,
        'title': '一键处理',
        'description': '毫秒级处理速度\n实时预览处理效果',
        'route': AppRouterConfig.processing,
      },
    ];

    return Column(
      children: [
        // 标题区域
        _buildHeader(theme),
        SizedBox(height: AppTheme.spacingXL),

        // 工作流程步骤
        LayoutBuilder(
          builder: (context, constraints) {
            if (constraints.maxWidth > 600) {
              // 桌面布局 - 横向排列
              return Row(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: steps.asMap().entries.map((entry) {
                  final index = entry.key;
                  final step = entry.value;
                  return Expanded(
                    child: Row(
                      children: [
                        Expanded(
                          child: _WorkflowStep(
                            number: step['number']! as String,
                            icon: step['icon']! as IconData,
                            title: step['title']! as String,
                            description: step['description']! as String,
                            route: step['route']! as String,
                          ),
                        ),
                        if (index < steps.length - 1)
                          Padding(
                            padding: EdgeInsets.symmetric(
                              horizontal: AppTheme.spacingM,
                            ),
                            child: Icon(
                              Icons.arrow_forward,
                              color: theme.dividerColor,
                              size: 24,
                            ),
                          ),
                      ],
                    ),
                  );
                }).toList(),
              );
            } else {
              // 移动布局 - 纵向排列
              return Column(
                children: steps
                    .map(
                      (step) => Padding(
                        padding: EdgeInsets.only(bottom: AppTheme.spacingL),
                        child: _WorkflowStep(
                          number: step['number']! as String,
                          icon: step['icon']! as IconData,
                          title: step['title']! as String,
                          description: step['description']! as String,
                          route: step['route']! as String,
                        ),
                      ),
                    )
                    .toList(),
              );
            }
          },
        ),
      ],
    );
  }

  /// 构建标题区域
  Widget _buildHeader(ThemeData theme) => Column(
    children: [
      Text(
        '强大的功能生态',
        style: theme.textTheme.headlineMedium?.copyWith(
          fontWeight: FontWeight.w700,
        ),
      ),
      SizedBox(height: AppTheme.spacingM),
      Text(
        '从输入到输出，每一步都精心设计',
        style: theme.textTheme.bodyLarge?.copyWith(
          color: theme.textTheme.bodyMedium?.color,
        ),
        textAlign: TextAlign.center,
      ),
    ],
  );
}

/// 单个工作流程步骤组件
class _WorkflowStep extends StatelessWidget {
  const _WorkflowStep({
    required this.number,
    required this.icon,
    required this.title,
    required this.description,
    required this.route,
  });

  final String number;
  final IconData icon;
  final String title;
  final String description;
  final String route;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return InkWell(
      onTap: () => context.go(route),
      borderRadius: BorderRadius.circular(AppTheme.spacingXL),
      child: Container(
        padding: EdgeInsets.all(AppTheme.spacingXL),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(AppTheme.spacingXL),
          boxShadow: AppTheme.getShadow(2),
        ),
        child: Column(
          children: [
            // 步骤编号
            Align(
              alignment: Alignment.topRight,
              child: Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFeff6ff), Color(0xFFdbeafe)],
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Center(
                  child: Text(
                    number,
                    style: TextStyle(
                      color: Color(0xFF3b82f6),
                      fontWeight: FontWeight.w700,
                      fontSize: 12,
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(height: AppTheme.spacingM),

            // 图标
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF3b82f6), Color(0xFF2563eb)],
                ),
                borderRadius: BorderRadius.circular(AppTheme.radiusL),
              ),
              child: Icon(icon, color: Colors.white, size: 28),
            ),
            SizedBox(height: AppTheme.spacingL),

            // 标题
            Text(
              title,
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w700,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: AppTheme.spacingM),

            // 描述
            Text(
              description,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.textTheme.bodyMedium?.color,
                height: 1.6,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
