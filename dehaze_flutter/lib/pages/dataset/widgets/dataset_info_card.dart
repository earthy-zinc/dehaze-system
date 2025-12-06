import 'package:flutter/material.dart';
import '../../../utils/responsive_utils.dart';
import '../models/dataset_model.dart';

/// 数据集信息卡片组件
///
/// 与设计稿 dataset.css 的 dataset-info-card 样式对应
/// 展示数据集详细信息和统计数据
class DatasetInfoCard extends StatelessWidget {
  const DatasetInfoCard({required this.dataset, super.key});

  final DatasetModel dataset;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isMobile = ResponsiveUtils.isMobile(context);

    // 设计稿颜色 - teal 到 cyan 渐变
    const tealColor = Color(0xFF14B8A6);
    const cyanColor = Color(0xFF06B6D4);

    return Container(
      padding: EdgeInsets.all(isMobile ? 16 : 20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [tealColor, cyanColor],
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
        ),
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: tealColor.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 标题
          Text(
            dataset.name,
            style: theme.textTheme.titleLarge?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w700,
              fontSize: isMobile ? 18 : 20,
            ),
          ),

          const SizedBox(height: 8),

          // 描述
          Text(
            dataset.description ?? '暂无描述',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: Colors.white.withValues(alpha: 0.9),
              height: 1.6,
              fontSize: 14,
            ),
          ),

          const SizedBox(height: 16),

          // 统计信息网格 - 与设计稿 stats-grid 对应
          _buildStatsGrid(context, isMobile),
        ],
      ),
    );
  }

  Widget _buildStatsGrid(BuildContext context, bool isMobile) {
    final stats = [
      _StatData('总计', dataset.totalImages),
      _StatData('有雾', dataset.foggyCount),
      _StatData('无雾', dataset.clearCount),
      _StatData('标注', dataset.annotatedCount),
    ];

    return Row(
      children: stats.map((stat) {
        return Expanded(
          child: _buildStatBox(context, stat.label, stat.value, isMobile),
        );
      }).toList(),
    );
  }

  Widget _buildStatBox(
    BuildContext context,
    String label,
    int value,
    bool isMobile,
  ) =>
      Container(
        margin: const EdgeInsets.symmetric(horizontal: 4),
        padding: EdgeInsets.symmetric(
          vertical: isMobile ? 8 : 12,
          horizontal: 4,
        ),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          children: [
            Text(
              '$value',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w700,
                fontSize: isMobile ? 18 : 24,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.8),
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
}

class _StatData {
  const _StatData(this.label, this.value);
  final String label;
  final int value;
}
