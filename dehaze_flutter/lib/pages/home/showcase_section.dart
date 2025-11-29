import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// 效果展示区组件
///
/// 展示去雾前后的对比效果
class ShowcaseSection extends StatelessWidget {
  const ShowcaseSection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: const EdgeInsets.all(AppTheme.spacingXL),
      child: Column(
        children: [
          // 标题区域
          const SizedBox(height: AppTheme.spacingXXL),

          _buildHeader(theme),

          const SizedBox(height: 64),

          // 效果对比图片
          _buildComparisonImage(context, theme),
        ],
      ),
    );
  }

  /// 构建标题区域
  Widget _buildHeader(ThemeData theme) => Column(
    children: [
      Text(
        '一键去雾，效果显著',
        style: theme.textTheme.headlineLarge?.copyWith(
          fontWeight: FontWeight.w700,
        ),
      ),

      const SizedBox(height: AppTheme.spacingM),

      Text(
        '智能算法自动识别雾霾程度，精准还原图像细节',
        style: theme.textTheme.titleLarge?.copyWith(
          fontSize: 20,
          color: Colors.grey[700],
          height: 1.6,
        ),
        textAlign: TextAlign.center,
      ),
    ],
  );

  /// 构建对比图片区域
  Widget _buildComparisonImage(BuildContext context, ThemeData theme) =>
      Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(AppTheme.spacingXL),
          boxShadow: AppTheme.getShadow(4),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(AppTheme.spacingXL),
          child: Stack(
            children: [
              // 实际的对比图片占位符
              Container(
                height: 600,
                width: double.infinity,
                color: theme.colorScheme.surfaceContainerHighest,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final halfWidth = constraints.maxWidth / 2;
                    return Stack(
                      children: [
                        // 左侧 - 去雾前图片占位符
                        Positioned(
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: halfWidth - 1,
                          child: Container(
                            color: Colors.grey.shade300,
                            child: const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.cloud,
                                    size: 48,
                                    color: Colors.grey,
                                  ),
                                  SizedBox(height: 8),
                                  Text(
                                    '去雾前',
                                    style: TextStyle(
                                      color: Colors.grey,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        // 右侧 - 去雾后图片占位符
                        Positioned(
                          right: 0,
                          top: 0,
                          bottom: 0,
                          width: halfWidth - 1,
                          child: Container(
                            color: Colors.blue.shade100,
                            child: const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.wb_sunny,
                                    size: 48,
                                    color: Colors.blue,
                                  ),
                                  SizedBox(height: 8),
                                  Text(
                                    '去雾后',
                                    style: TextStyle(
                                      color: Colors.blue,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        // 中间分割线
                        Center(
                          child: Container(
                            width: 2,
                            height: double.infinity,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
              // 底部标签
              Positioned(
                bottom: AppTheme.spacingL,
                left: 0,
                right: 0,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: AppTheme.spacingL,
                      vertical: AppTheme.spacingM,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.75),
                      borderRadius: BorderRadius.circular(50),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          '去雾前',
                          style: TextStyle(
                            color: Color(0xFFFBBF24),
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                          ),
                        ),
                        SizedBox(width: AppTheme.spacingM),
                        Text(
                          '→',
                          style: TextStyle(
                            color: Color(0xFF9CA3AF),
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                          ),
                        ),
                        SizedBox(width: AppTheme.spacingM),
                        Text(
                          '去雾后',
                          style: TextStyle(
                            color: Color(0xFF34D399),
                            fontWeight: FontWeight.w600,
                            fontSize: 14,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
}
