import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/responsive_utils.dart';
import '../models/dataset_model.dart';

/// 图片网格组件
///
/// 支持响应式布局，根据屏幕宽度自动调整列数
/// 与设计稿 dataset.css 的 waterfall-grid 保持一致
class ImageGrid extends StatelessWidget {
  const ImageGrid({
    required this.images,
    super.key,
    this.onImageTap,
    this.onLoadMore,
    this.hasMore = true,
    this.isLoading = false,
    this.asSliver = false,
  });

  final List<ImageModel> images;
  final void Function(ImageModel)? onImageTap;
  final VoidCallback? onLoadMore;
  final bool hasMore;
  final bool isLoading;
  /// 是否作为 Sliver 使用（用于 CustomScrollView）
  final bool asSliver;

  @override
  Widget build(BuildContext context) {
    if (images.isEmpty && !isLoading) {
      if (asSliver) {
        return SliverFillRemaining(child: _buildEmptyState(context));
      }
      return _buildEmptyState(context);
    }

    // Sliver 模式：直接使用 MediaQuery 获取屏幕宽度
    if (asSliver) {
      return _buildSliverGrid(context);
    }

    // 普通模式：使用 LayoutBuilder
    return LayoutBuilder(
      builder: (context, constraints) {
        final crossAxisCount = ResponsiveUtils.getGridCrossAxisCount(
          context,
          mobile: 2,
          tablet: 3,
          desktop: 4,
          largeDesktop: 5,
        );
        final spacing = ResponsiveUtils.getResponsiveSpacing(context);
        final gridDelegate = SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: crossAxisCount,
          crossAxisSpacing: spacing,
          mainAxisSpacing: spacing,
          childAspectRatio: 1,
        );

        return NotificationListener<ScrollNotification>(
          onNotification: _onScroll,
          child: GridView.builder(
            padding: EdgeInsets.all(spacing),
            gridDelegate: gridDelegate,
            itemCount: images.length + (hasMore && isLoading ? 1 : 0),
            itemBuilder: (context, index) {
              if (index == images.length && hasMore && isLoading) {
                return _buildLoadingItem();
              }
              final image = images[index];
              return _ImageCard(
                image: image,
                onTap: () => onImageTap?.call(image),
              );
            },
          ),
        );
      },
    );
  }

  /// 构建 Sliver 模式的网格
  Widget _buildSliverGrid(BuildContext context) {
    final crossAxisCount = ResponsiveUtils.getGridCrossAxisCount(
      context,
      mobile: 2,
      tablet: 3,
      desktop: 4,
      largeDesktop: 5,
    );
    final spacing = ResponsiveUtils.getResponsiveSpacing(context);
    final gridDelegate = SliverGridDelegateWithFixedCrossAxisCount(
      crossAxisCount: crossAxisCount,
      crossAxisSpacing: spacing,
      mainAxisSpacing: spacing,
      childAspectRatio: 1,
    );

    return SliverPadding(
      padding: EdgeInsets.all(spacing),
      sliver: SliverGrid(
        gridDelegate: gridDelegate,
        delegate: SliverChildBuilderDelegate(
          (context, index) {
            if (index == images.length && hasMore && isLoading) {
              return _buildLoadingItem();
            }
            final image = images[index];
            return _ImageCard(
              image: image,
              onTap: () => onImageTap?.call(image),
            );
          },
          childCount: images.length + (hasMore && isLoading ? 1 : 0),
        ),
      ),
    );
  }

  Widget _buildLoadingItem() => Card(
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
            color: Colors.grey[200],
          ),
          child: const Center(child: CircularProgressIndicator()),
        ),
      );

  Widget _buildEmptyState(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.image_outlined,
            size: 64,
            color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
          ),
          SizedBox(height: AppTheme.spacingM),
          Text(
            '暂无图片',
            style: theme.textTheme.titleLarge?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
            ),
          ),
        ],
      ),
    );
  }

  bool _onScroll(ScrollNotification scrollInfo) {
    if (!hasMore || isLoading) {
      return false;
    }

    if (scrollInfo is ScrollEndNotification &&
        scrollInfo.metrics.extentAfter < 200) {
      onLoadMore?.call();
    }

    return false;
  }
}

/// 单个图片卡片组件
///
/// 支持悬停效果，与设计稿 waterfall-item 样式对应
class _ImageCard extends StatefulWidget {
  const _ImageCard({
    required this.image,
    this.onTap,
  });

  final ImageModel image;
  final VoidCallback? onTap;

  @override
  State<_ImageCard> createState() => _ImageCardState();
}

class _ImageCardState extends State<_ImageCard>
    with SingleTickerProviderStateMixin {
  bool _isHovered = false;
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

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
            ..setEntry(0, 0, _isPressed ? 0.95 : 1.0)
            ..setEntry(1, 1, _isPressed ? 0.95 : 1.0),
          child: Card(
            elevation: _isHovered ? 8 : 2,
            shadowColor: Colors.black.withValues(alpha: _isHovered ? 0.12 : 0.08),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppTheme.radiusM),
            ),
            child: InkWell(
              onTap: widget.onTap,
              borderRadius: BorderRadius.circular(AppTheme.radiusM),
              child: Stack(
                children: [
                  // 图片
                  Positioned.fill(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(AppTheme.radiusM),
                      child: CachedNetworkImage(
                        imageUrl: widget.image.imageUrl,
                        fit: BoxFit.cover,
                        placeholder: (context, url) => Container(
                          color: theme.colorScheme.surfaceContainerHighest,
                          child:
                              const Center(child: CircularProgressIndicator()),
                        ),
                        errorWidget: (context, url, error) => Container(
                          color: theme.colorScheme.surfaceContainerHighest,
                          child: const Icon(
                            Icons.broken_image_outlined,
                            color: Colors.grey,
                            size: 32,
                          ),
                        ),
                      ),
                    ),
                  ),

                  // 类型标签 - 与设计稿 type-badge 对应
                  Positioned(
                    top: 8,
                    right: 8,
                    child: _TypeBadge(type: widget.image.imageType),
                  ),

                  // 文件名和信息（底部显示）
                  Positioned(
                    bottom: 0,
                    left: 0,
                    right: 0,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.only(
                          bottomLeft: Radius.circular(AppTheme.radiusM),
                          bottomRight: Radius.circular(AppTheme.radiusM),
                        ),
                        gradient: LinearGradient(
                          begin: Alignment.bottomCenter,
                          end: Alignment.topCenter,
                          colors: [
                            Colors.black.withValues(alpha: 0.7),
                            Colors.transparent,
                          ],
                        ),
                      ),
                      child: Text(
                        widget.image.filename,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// 类型标签组件
///
/// 与设计稿 type-badge 样式对应
class _TypeBadge extends StatelessWidget {
  const _TypeBadge({required this.type});

  final ImageType type;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: _getBackgroundColor(),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          type.displayName,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w500,
          ),
        ),
      );

  Color _getBackgroundColor() {
    switch (type) {
      case ImageType.hazy:
        return const Color(0xE66B7280); // rgba(107, 114, 128, 0.9)
      case ImageType.clear:
        return const Color(0xE63B82F6); // rgba(59, 130, 246, 0.9)
      case ImageType.dehazed:
        return const Color(0xE610B981); // rgba(16, 185, 129, 0.9)
    }
  }
}
