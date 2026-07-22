import 'package:flutter/material.dart';

import '../../core/network/api_config.dart';
import '../../theme/app_theme.dart';
import '../../widgets/dehaze_image.dart';

/// 效果展示区组件
///
/// 展示去雾前后的对比效果
class ShowcaseSection extends StatelessWidget {
  const ShowcaseSection({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingXL),
      child: Column(
        children: [
          // 标题区域
          SizedBox(height: AppTheme.spacingXXL),

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

      SizedBox(height: AppTheme.spacingM),

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
  ///
  /// 使用 NH-HAZE-2023 数据集真实样本展示去雾前后对比
  Widget _buildComparisonImage(BuildContext context, ThemeData theme) {
    // NH-HAZE-2023 数据集真实样本：001 号样本的雾图与清晰图
    final beforeUrl =
        '${ApiConfig.datasetBaseUrl}/NH-HAZE-2023/hazy/001.JPG';
    final afterUrl =
        '${ApiConfig.datasetBaseUrl}/NH-HAZE-2023/clean/001.JPG';

    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(AppTheme.spacingXL),
        boxShadow: AppTheme.getShadow(4),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppTheme.spacingXL),
        child: Stack(
          children: [
            // 真实去雾前后对比图片
            SizedBox(
              height: 600,
              width: double.infinity,
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final halfWidth = constraints.maxWidth / 2;
                  return Stack(
                    children: [
                      // 左侧 - 去雾前（雾图）
                      Positioned(
                        left: 0,
                        top: 0,
                        bottom: 0,
                        width: halfWidth - 1,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            DehazeImage(
                              url: beforeUrl,
                              fit: BoxFit.cover,
                              placeholderIcon: Icons.cloud,
                            ),
                            Positioned(
                              top: AppTheme.spacingM,
                              left: AppTheme.spacingM,
                              child: _buildSideLabel('去雾前', const Color(0xFFFBBF24)),
                            ),
                          ],
                        ),
                      ),
                      // 右侧 - 去雾后（清晰图）
                      Positioned(
                        right: 0,
                        top: 0,
                        bottom: 0,
                        width: halfWidth - 1,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            DehazeImage(
                              url: afterUrl,
                              fit: BoxFit.cover,
                              placeholderIcon: Icons.wb_sunny,
                            ),
                            Positioned(
                              top: AppTheme.spacingM,
                              right: AppTheme.spacingM,
                              child: _buildSideLabel('去雾后', const Color(0xFF34D399)),
                            ),
                          ],
                        ),
                      ),
                      // 中间分割线
                      Center(
                        child: Container(
                          width: 2,
                          height: double.infinity,
                          color: Colors.white.withValues(alpha: 0.7),
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
                  padding: EdgeInsets.symmetric(
                    horizontal: AppTheme.spacingL,
                    vertical: AppTheme.spacingM,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.75),
                    borderRadius: BorderRadius.circular(50),
                  ),
                  child: Row(
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

  /// 构建图片角标标签
  Widget _buildSideLabel(String text, Color color) => Container(
        padding: EdgeInsets.symmetric(
          horizontal: AppTheme.spacingS,
          vertical: AppTheme.spacingXS,
        ),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(AppTheme.radiusS),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.w600,
            fontSize: 13,
          ),
        ),
      );
}
