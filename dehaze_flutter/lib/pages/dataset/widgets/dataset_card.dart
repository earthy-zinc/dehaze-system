import 'package:flutter/material.dart';
import '../../../core/network/api_config.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/responsive_utils.dart';
import '../models/dataset_model.dart';

/// 数据集卡片组件
///
/// 支持悬停效果、点击缩放动画
class DatasetCard extends StatefulWidget {
  const DatasetCard({required this.dataset, super.key, this.onTap});

  final DatasetModel dataset;
  final VoidCallback? onTap;

  @override
  State<DatasetCard> createState() => _DatasetCardState();
}

class _DatasetCardState extends State<DatasetCard> {
  bool _isHovered = false;
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isMobile = ResponsiveUtils.isMobile(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTapDown: (_) => setState(() => _isPressed = true),
        onTapUp: (_) => setState(() => _isPressed = false),
        onTapCancel: () => setState(() => _isPressed = false),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          transform: Matrix4.identity()
            ..setEntry(1, 3, _isHovered ? -2.0 : 0.0)
            ..setEntry(0, 0, _isPressed ? 0.98 : 1.0)
            ..setEntry(1, 1, _isPressed ? 0.98 : 1.0),
          child: Card(
            elevation: _isHovered ? 12 : 2,
            shadowColor: Colors.black.withValues(alpha: _isHovered ? 0.12 : 0.08),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            clipBehavior: Clip.antiAlias,
            child: InkWell(
              onTap: widget.onTap,
              child: isMobile
                  ? _buildMobileLayout(theme)
                  : _buildDesktopLayout(theme),
            ),
          ),
        ),
      ),
    );
  }

  /// 移动端布局 - 纵向排列
  Widget _buildMobileLayout(ThemeData theme) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildThumbnail(height: 120),
          _buildContent(theme),
        ],
      );

  /// 桌面端布局 - 横向排列
  Widget _buildDesktopLayout(ThemeData theme) => Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildThumbnail(width: 128, height: 128),
          Expanded(child: _buildContent(theme)),
        ],
      );

  /// 构建缩略图头部
  ///
  /// 优先使用数据集 path 拼接真实样本图（hazy/001.JPG）作为封面；
  /// path 为空或图片加载失败时，fallback 到渐变图标占位。
  Widget _buildThumbnail({double? width, double? height}) {
    final path = widget.dataset.path;
    final hasSample = path != null && path.isNotEmpty;

    if (!hasSample) {
      return _buildGradientFallback(width: width, height: height);
    }

    final sampleUrl = '${ApiConfig.datasetBaseUrl}/$path/hazy/001.JPG';
    return SizedBox(
      width: width ?? double.infinity,
      height: height,
      child: Image.network(
        sampleUrl,
        fit: BoxFit.cover,
        gaplessPlayback: true,
        loadingBuilder: (context, child, progress) {
          if (progress == null) return child;
          // 加载中显示渐变图标占位
          return _buildGradientFallback(width: width, height: height);
        },
        errorBuilder: (_, _, _) =>
            _buildGradientFallback(width: width, height: height),
      ),
    );
  }

  /// 渐变图标占位（path 为空或图片加载失败时的 fallback）
  Widget _buildGradientFallback({double? width, double? height}) => Container(
        width: width ?? double.infinity,
        height: height,
        decoration: BoxDecoration(
          gradient: AppTheme.getSecondaryGradient(),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                widget.dataset.hasChildren
                    ? Icons.folder_outlined
                    : Icons.storage_outlined,
                color: Colors.white,
                size: 40,
              ),
              if (widget.dataset.hasChildren) ...[
                const SizedBox(height: 4),
                Text(
                  '${widget.dataset.children.length} 个子集',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.8),
                    fontSize: 12,
                  ),
                ),
              ],
            ],
          ),
        ),
      );

  /// 构建内容区域
  Widget _buildContent(ThemeData theme) => Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 标题
            Text(
              widget.dataset.name,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: const Color(0xFF1F2937),
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),

            const SizedBox(height: 8),

            // 描述
            Text(
              widget.dataset.description ?? '暂无描述',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: const Color(0xFF6B7280),
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),

            const SizedBox(height: 12),

            // 统计信息
            Row(
              children: [
                if (widget.dataset.type != null) ...[
                  _buildStatItem(
                    context,
                    Icons.category_outlined,
                    widget.dataset.type!,
                    const Color(0xFF14B8A6),
                  ),
                  const SizedBox(width: 16),
                ],
                _buildStatItem(
                  context,
                  Icons.access_time_outlined,
                  _formatDate(widget.dataset.createTime),
                  const Color(0xFF9CA3AF),
                ),
              ],
            ),
          ],
        ),
      );

  Widget _buildStatItem(
    BuildContext context,
    IconData icon,
    String text,
    Color color,
  ) =>
      Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 4),
          Text(
            text,
            style: const TextStyle(
              fontSize: 13,
              color: Color(0xFF9CA3AF),
            ),
          ),
        ],
      );

  String _formatDate(String dateStr) {
    try {
      final date = DateTime.parse(dateStr);
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
    } catch (_) {
      return dateStr;
    }
  }
}
