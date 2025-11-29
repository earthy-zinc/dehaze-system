import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 算法优势区域组件
///
/// 展示多种算法的优势和特性
class AlgorithmSection extends StatelessWidget {
  const AlgorithmSection({super.key});

  @override
  Widget build(BuildContext context) => Container(
    decoration: const BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Color(0xFF1e3a8a), Color(0xFF3b82f6)],
      ),
    ),
    padding: const EdgeInsets.all(AppTheme.spacingXL),
    child: LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth > 800) {
          // 桌面布局
          return Row(
            children: [
              Expanded(flex: 1, child: _buildAlgorithmText(context)),
              const SizedBox(width: AppTheme.spacingXXL),
              Expanded(flex: 1, child: _buildAlgorithmVisual(context)),
            ],
          );
        } else {
          // 移动布局
          return Column(
            children: [
              _buildAlgorithmText(context),
              const SizedBox(height: AppTheme.spacingXL),
              _buildAlgorithmVisual(context),
            ],
          );
        }
      },
    ),
  );

  /// 构建算法文本内容
  Widget _buildAlgorithmText(BuildContext context) {
    final theme = Theme.of(context);
    final features = [
      '智能推荐最适合的去雾算法',
      '实时对比不同算法的处理效果',
      '毫秒级处理速度，即时查看结果',
      '支持批量处理和参数自定义',
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '多算法智能选择',
          style: theme.textTheme.headlineMedium?.copyWith(
            color: Colors.white,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: AppTheme.spacingM),
        Text(
          '支持DCP、AOD-Net、DehazeNet等多种先进算法',
          style: theme.textTheme.bodyLarge?.copyWith(
            color: Colors.white.withValues(alpha: 0.8),
          ),
        ),
        const SizedBox(height: AppTheme.spacingXL),

        // 特性列表
        ...features.map(
          (feature) => Padding(
            padding: const EdgeInsets.only(bottom: AppTheme.spacingM),
            child: Row(
              children: [
                const Icon(
                  Icons.check_circle,
                  color: Color(0xFF34d399),
                  size: 20,
                ),
                const SizedBox(width: AppTheme.spacingM),
                Expanded(
                  child: Text(
                    feature,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: Colors.white.withValues(alpha: 0.95),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: AppTheme.spacingXL),

        // 了解更多按钮
        ElevatedButton.icon(
          onPressed: () => context.go(AppRouterConfig.algorithm),
          icon: const Icon(Icons.arrow_forward, size: 16),
          label: const Text('了解更多算法详情'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.white,
            foregroundColor: const Color(0xFF3b82f6),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppTheme.radiusL),
            ),
          ),
        ),
      ],
    );
  }

  /// 构建算法视觉展示
  Widget _buildAlgorithmVisual(BuildContext context) => Container(
    height: 240,
    decoration: BoxDecoration(
      borderRadius: BorderRadius.circular(AppTheme.spacingXL),
      boxShadow: const [
        BoxShadow(
          offset: Offset(0, 20),
          blurRadius: 60,
          color: Color(0x4D000000),
        ),
      ],
    ),
    child: ClipRRect(
      borderRadius: BorderRadius.circular(AppTheme.spacingXL),
      child: Stack(
        children: [
          // 背景渐变
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Colors.white.withValues(alpha: 0.1),
                  Colors.white.withValues(alpha: 0.05),
                ],
              ),
            ),
          ),
          // 中心图标和文字
          const Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.psychology, size: 80, color: Colors.white),
                SizedBox(height: AppTheme.spacingL),
                Text(
                  'AI算法引擎',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                SizedBox(height: AppTheme.spacingM),
                Text(
                  '深度学习驱动',
                  style: TextStyle(color: Colors.white70, fontSize: 14),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );
}
