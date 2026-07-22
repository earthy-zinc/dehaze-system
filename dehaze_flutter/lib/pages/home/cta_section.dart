import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 最终CTA区域组件
///
/// 页面底部的行动号召区域
class CTASection extends StatelessWidget {
  const CTASection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingXL),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            theme.colorScheme.surface,
            theme.colorScheme.surfaceContainerHighest,
          ],
        ),
      ),
      child: Column(
        children: [
          Text(
            '准备好体验专业级图像去雾了吗？',
            style: theme.textTheme.headlineMedium?.copyWith(
              fontWeight: FontWeight.w700,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: AppTheme.spacingM),

          Text(
            '立即开始，让您的图像重获清晰',
            style: theme.textTheme.bodyLarge?.copyWith(
              color: theme.textTheme.bodyMedium?.color,
            ),
            textAlign: TextAlign.center,
          ),
          SizedBox(height: AppTheme.spacingXL),

          // CTA按钮
          _buildCTAButton(context),
          SizedBox(height: AppTheme.spacingXL),

          // 额外的选项链接
          _buildAdditionalLinks(context),
        ],
      ),
    );
  }

  /// 构建主要CTA按钮
  Widget _buildCTAButton(BuildContext context) => ElevatedButton.icon(
    onPressed: () => context.go(AppRouterConfig.imageInput),
    icon: Icon(Icons.arrow_forward, size: 20),
    label: const Text('开始使用'),
    style: ElevatedButton.styleFrom(
      backgroundColor: const Color(0xFF3b82f6),
      foregroundColor: Colors.white,
      padding: EdgeInsets.symmetric(
        horizontal: AppTheme.spacingXXL * 2,
        vertical: AppTheme.spacingL,
      ),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusXL),
      ),
      elevation: 8,
    ),
  );

  /// 构建额外选项链接
  Widget _buildAdditionalLinks(BuildContext context) => Wrap(
    alignment: WrapAlignment.center,
    spacing: AppTheme.spacingXL,
    runSpacing: AppTheme.spacingM,
    children: [
      TextButton.icon(
        onPressed: () => context.go(AppRouterConfig.dataset),
        icon: Icon(Icons.dataset, size: 16),
        label: const Text('查看示例'),
        style: TextButton.styleFrom(
          foregroundColor: Theme.of(context).colorScheme.primary,
        ),
      ),
      TextButton.icon(
        onPressed: () => context.go(AppRouterConfig.algorithm),
        icon: Icon(Icons.science, size: 16),
        label: const Text('了解算法'),
        style: TextButton.styleFrom(
          foregroundColor: Theme.of(context).colorScheme.primary,
        ),
      ),
    ],
  );
}
