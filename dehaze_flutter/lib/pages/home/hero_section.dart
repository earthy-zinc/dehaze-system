import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// Hero Section - 英雄区域组件
class HeroSection extends StatelessWidget {
  const HeroSection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: AppTheme.spacingM,
        vertical: AppTheme.spacingXXL,
      ),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [colorScheme.surface, colorScheme.surfaceContainerHighest],
        ),
      ),
      child: Column(
        children: [
          const SizedBox(height: AppTheme.spacingXL),

          // 主标题 - 使用渐变色
          ShaderMask(
            shaderCallback: (bounds) => const LinearGradient(
              colors: [Color(0xFF1e40af), Color(0xFF3b82f6), Color(0xFF60a5fa)],
            ).createShader(bounds),
            child: Text(
              '图像去雾',
              style: theme.textTheme.headlineLarge?.copyWith(
                fontSize: 58,
                color: Colors.white,
              ),
            ),
          ),

          const SizedBox(height: AppTheme.spacingS),

          // 副标题
          Text(
            '专业级图像处理系统',
            style: theme.textTheme.headlineMedium?.copyWith(fontSize: 36),
          ),

          const SizedBox(height: AppTheme.spacingXL),

          // 描述文本
          Text(
            '采用先进的深度学习算法，一键还原清晰视界\n从图像输入到效果评估的完整闭环体验',
            style: theme.textTheme.titleLarge?.copyWith(
              fontSize: 20,
              color: Colors.grey[700],
              height: 1.6,
            ),
            textAlign: TextAlign.center,
          ),

          const SizedBox(height: 36),

          // CTA按钮组
          _buildCTAButtons(context),

          const SizedBox(height: 64),
        ],
      ),
    );
  }

  /// 构建CTA按钮组
  Widget _buildCTAButtons(BuildContext context) => Wrap(
    alignment: WrapAlignment.center,
    spacing: AppTheme.spacingM,
    runSpacing: AppTheme.spacingM,
    children: [
      // 主要按钮 - 立即开始
      ElevatedButton.icon(
        onPressed: () {
          // 使用go_router导航到图像输入页面
          context.go(AppRouterConfig.imageInput);
        },
        icon: const Icon(Icons.arrow_forward, size: 26),
        label: const Text(
          '立即开始',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF3b82f6),
          foregroundColor: Colors.white,

          padding: const EdgeInsets.symmetric(
            horizontal: AppTheme.spacingXL,
            vertical: AppTheme.spacingL,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
          ),
          elevation: 4,
        ),
      ),

      // 次要按钮 - 浏览数据集
      OutlinedButton.icon(
        onPressed: () {
          // 使用go_router导航到数据集页面
          context.go(AppRouterConfig.dataset);
        },
        icon: const Icon(Icons.dataset, size: 26),
        label: const Text(
          '浏览数据集',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
        ),
        style: OutlinedButton.styleFrom(
          padding: const EdgeInsets.symmetric(
            horizontal: AppTheme.spacingXL,
            vertical: AppTheme.spacingL,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
          ),
        ),
      ),
    ],
  );
}
