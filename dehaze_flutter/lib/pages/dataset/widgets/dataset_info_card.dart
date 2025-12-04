import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

class DatasetInfoCard extends StatelessWidget {
  const DatasetInfoCard({required this.dataset, super.key});

  final DatasetModel dataset;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingL),
      decoration: BoxDecoration(
        gradient: AppTheme.getPrimaryGradient(),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 标题
          Text(
            dataset.name,
            style: theme.textTheme.headlineSmall?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w700,
            ),
          ),

          SizedBox(height: AppTheme.spacingS),

          // 描述
          Text(
            dataset.description ?? '暂无描述',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: Colors.white.withValues(alpha: 0.9),
            ),
          ),

          SizedBox(height: AppTheme.spacingL),

          // 统计信息网格
          Row(
            children: [
              _buildStatBox(
                context,
                '总计',
                '${dataset.totalImages}',
                Colors.white,
              ),
              SizedBox(width: AppTheme.spacingM),
              _buildStatBox(
                context,
                '有雾',
                '${dataset.foggyCount}',
                Colors.white,
              ),
              SizedBox(width: AppTheme.spacingM),
              _buildStatBox(
                context,
                '无雾',
                '${dataset.clearCount}',
                Colors.white,
              ),
              SizedBox(width: AppTheme.spacingM),
              _buildStatBox(
                context,
                '标注',
                '${dataset.annotatedCount}',
                Colors.white,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatBox(
    BuildContext context,
    String label,
    String value,
    Color textColor,
  ) => Expanded(
    child: Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.headlineMedium?.copyWith(
            color: textColor,
            fontWeight: FontWeight.w700,
          ),
        ),
        SizedBox(height: AppTheme.spacingXS),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: textColor.withValues(alpha: 0.8),
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    ),
  );
}
