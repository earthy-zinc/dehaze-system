import 'package:flutter/material.dart';
import '../../../utils/responsive_utils.dart';
import '../../../models/dataset_model.dart';

/// 数据集信息卡片组件
///
/// 展示数据集详细信息和统计数据
class DatasetInfoCard extends StatelessWidget {
  const DatasetInfoCard({required this.dataset, super.key});

  final Dataset dataset;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isMobile = ResponsiveUtils.isMobile(context);

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

          // 信息网格
          _buildInfoGrid(context, isMobile),
        ],
      ),
    );
  }

  Widget _buildInfoGrid(BuildContext context, bool isMobile) {
    final info = [
      _InfoData('类型', dataset.type),
      _InfoData('创建时间', dataset.createTime ?? '未知'),
      if (dataset.children != null && dataset.children!.isNotEmpty)
        _InfoData('子数据集', '${dataset.children!.length}'),
    ];

    return Wrap(
      spacing: 16,
      runSpacing: 8,
      children: info.map((item) => _buildInfoItem(item, isMobile)).toList(),
    );
  }

  Widget _buildInfoItem(_InfoData item, bool isMobile) => Container(
        padding: EdgeInsets.symmetric(
          vertical: isMobile ? 6 : 8,
          horizontal: 12,
        ),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '${item.label}:',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 4),
            Text(
              item.value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      );
}

class _InfoData {
  const _InfoData(this.label, this.value);
  final String label;
  final String value;
}
