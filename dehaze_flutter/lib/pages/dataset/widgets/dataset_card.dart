import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

class DatasetCard extends StatelessWidget {
  const DatasetCard({required this.dataset, super.key, this.onTap});

  final DatasetModel dataset;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return Card(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 缩略图
            ClipRRect(
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(AppTheme.radiusL),
                bottomLeft: Radius.circular(AppTheme.radiusL),
              ),
              child: Container(
                width: 128,
                height: 128,
                decoration: BoxDecoration(
                  gradient: AppTheme.getPrimaryGradient(),
                ),
                child: Image.network(
                  dataset.thumbnail,
                  width: 128,
                  height: 128,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => Icon(
                    Icons.storage_outlined,
                    color: Colors.white,
                    size: 48,
                  ),
                ),
              ),
            ),

            // 内容区域
            Expanded(
              child: Padding(
                padding: EdgeInsets.all(AppTheme.spacingM),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 标题
                    Text(
                      dataset.name,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),

                    SizedBox(height: AppTheme.spacingS),

                    // 描述
                    Text(
                      dataset.description ?? '暂无描述',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),

                    SizedBox(height: AppTheme.spacingM),

                    // 统计信息
                    Row(
                      children: [
                        _buildStatItem(
                          context,
                          Icons.image_outlined,
                          '${dataset.totalImages}',
                          colorScheme.primary,
                        ),
                        SizedBox(width: AppTheme.spacingL),
                        _buildStatItem(
                          context,
                          Icons.access_time_outlined,
                          _formatDate(dataset.createdAt),
                          theme.colorScheme.onSurfaceVariant,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatItem(
    BuildContext context,
    IconData icon,
    String text,
    Color color,
  ) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(icon, size: 16, color: color),
      SizedBox(width: AppTheme.spacingXS),
      Text(
        text,
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
          color: Theme.of(context).colorScheme.onSurfaceVariant,
        ),
      ),
    ],
  );

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final difference = now.difference(date);

    if (difference.inDays == 0) {
      return '今天';
    } else if (difference.inDays == 1) {
      return '昨天';
    } else if (difference.inDays < 7) {
      return '${difference.inDays}天前';
    } else {
      return '${date.month}/${date.day}';
    }
  }
}
