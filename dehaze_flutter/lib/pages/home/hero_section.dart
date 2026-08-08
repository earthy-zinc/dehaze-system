import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// Hero Section - 英雄区域组件
class HeroSection extends StatelessWidget {
  const HeroSection({super.key});

  /// 响应式字体大小计算
  double _getResponsiveFontSize(
    BuildContext context,
    double desktopSize,
    double mobileSize,
  ) {
    final width = MediaQuery.of(context).size.width;
    if (width > 768) return desktopSize;
    if (width > 480) return desktopSize * 0.8;
    return mobileSize;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Container(
      padding: EdgeInsets.symmetric(
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
          SizedBox(height: AppTheme.spacingXL),

          // 主标题 - 使用渐变色（与设计稿一致）
          ShaderMask(
            shaderCallback: (bounds) => const LinearGradient(
              colors: AppTheme.heroGradient,
            ).createShader(bounds),
            child: Text(
              '图像去雾',
              style: theme.textTheme.headlineLarge?.copyWith(
                fontSize: _getResponsiveFontSize(context, 58, 42),
                color: Colors.white,
              ),
            ),
          ),

          SizedBox(height: AppTheme.spacingS),

          // 副标题
          Text(
            '专业级图像处理系统',
            style: theme.textTheme.headlineMedium?.copyWith(
              fontSize: _getResponsiveFontSize(context, 36, 24),
            ),
          ),

          SizedBox(height: AppTheme.spacingXL),

          // 描述文本
          Text(
            '采用先进的深度学习算法，一键还原清晰视界\n从图像输入到效果评估的完整闭环体验',
            style: theme.textTheme.titleLarge?.copyWith(
              fontSize: _getResponsiveFontSize(context, 20, 16),
              color: Colors.grey[700],
              height: 1.6,
            ),
            textAlign: TextAlign.center,
          ),

          SizedBox(height: 36),

          // CTA按钮组
          _buildCTAButtons(context),

          SizedBox(height: 64),
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
          context.go(AppRouterConfig.imageInput);
        },
        icon: Icon(Icons.auto_fix_high, size: 26),
        label: Text(
          '开始去雾',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.brandBlue,
          foregroundColor: Colors.white,

          padding: EdgeInsets.symmetric(
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
        icon: Icon(Icons.dataset, size: 26),
        label: Text(
          '浏览数据集',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
        ),
        style: OutlinedButton.styleFrom(
          padding: EdgeInsets.symmetric(
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
