import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';

import '../models/image_input_model.dart';

/// 历史记录卡片
///
/// 显示历史记录的缩略图、文件名、时间和算法信息
class HistoryItemCard extends StatefulWidget {
  const HistoryItemCard({
    required this.record,
    super.key,
    this.onTap,
    this.onReprocess,
    this.onDelete,
  });

  final HistoryRecordModel record;
  final VoidCallback? onTap;
  final VoidCallback? onReprocess;
  final VoidCallback? onDelete;

  @override
  State<HistoryItemCard> createState() => _HistoryItemCardState();
}

class _HistoryItemCardState extends State<HistoryItemCard> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: Dismissible(
        key: Key(widget.record.id),
        direction: DismissDirection.endToStart,
        onDismissed: (_) => widget.onDelete?.call(),
        background: Container(
          alignment: Alignment.centerRight,
          padding: const EdgeInsets.only(right: 16),
          decoration: BoxDecoration(
            color: theme.colorScheme.error,
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Icon(
            Icons.delete_outline,
            color: Colors.white,
          ),
        ),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: _isHovered ? 0.1 : 0.05),
                blurRadius: _isHovered ? 12 : 6,
                offset: Offset(0, _isHovered ? 4 : 2),
              ),
            ],
          ),
          child: InkWell(
            onTap: widget.onTap,
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  // 缩略图
                  _buildThumbnail(theme),

                  const SizedBox(width: 12),

                  // 信息
                  Expanded(
                    child: _buildInfo(theme),
                  ),

                  // 操作按钮
                  _buildActions(theme),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildThumbnail(ThemeData theme) => ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: SizedBox(
          width: 64,
          height: 64,
          child: CachedNetworkImage(
            imageUrl: widget.record.originalThumbnail,
            fit: BoxFit.cover,
            placeholder: (context, url) => Container(
              color: theme.colorScheme.surfaceContainerHighest,
              child: const Center(
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
            errorWidget: (context, url, error) => Container(
              color: theme.colorScheme.surfaceContainerHighest,
              child: Icon(
                Icons.image_outlined,
                size: 24,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ),
      );

  Widget _buildInfo(ThemeData theme) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 文件名
          Text(
            widget.record.filename,
            style: theme.textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: theme.colorScheme.onSurface,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          // 时间
          Text(
            _formatTime(widget.record.timestamp),
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          // 算法名称
          if (widget.record.algorithmName != null)
            Row(
              children: [
                Icon(
                  Icons.auto_awesome_outlined,
                  size: 14,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    widget.record.algorithmName!,
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: theme.colorScheme.primary,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
        ],
      );

  Widget _buildActions(ThemeData theme) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 重新处理
          IconButton(
            onPressed: widget.onReprocess,
            icon: const Icon(Icons.refresh),
            iconSize: 20,
            tooltip: '重新处理',
            style: IconButton.styleFrom(
              foregroundColor: theme.colorScheme.primary,
            ),
          ),
          // 删除
          IconButton(
            onPressed: widget.onDelete,
            icon: const Icon(Icons.delete_outline),
            iconSize: 20,
            tooltip: '删除',
            style: IconButton.styleFrom(
              foregroundColor: theme.colorScheme.error,
            ),
          ),
        ],
      );

  String _formatTime(DateTime timestamp) {
    final now = DateTime.now();
    final diff = now.difference(timestamp);

    if (diff.inMinutes < 1) {
      return '刚刚';
    } else if (diff.inHours < 1) {
      return '${diff.inMinutes}分钟前';
    } else if (diff.inDays < 1) {
      return '${diff.inHours}小时前';
    } else if (diff.inDays < 7) {
      return '${diff.inDays}天前';
    } else {
      return '${timestamp.month}/${timestamp.day}';
    }
  }
}
