import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

class ImageGrid extends StatelessWidget {
  const ImageGrid({
    required this.images,
    super.key,
    this.onImageTap,
    this.onLoadMore,
    this.hasMore = true,
    this.isLoading = false,
  });

  final List<ImageModel> images;
  final Function(ImageModel)? onImageTap;
  final VoidCallback? onLoadMore;
  final bool hasMore;
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    if (images.isEmpty && !isLoading) {
      return _buildEmptyState(context);
    }

    return NotificationListener<ScrollNotification>(
      onNotification: _onScroll,
      child: GridView.builder(
        padding: EdgeInsets.all(AppTheme.spacingS),
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3,
          crossAxisSpacing: AppTheme.spacingS,
          mainAxisSpacing: AppTheme.spacingS,
          childAspectRatio: 1,
        ),
        itemCount: images.length + (hasMore && isLoading ? 1 : 0),
        itemBuilder: (context, index) {
          if (index == images.length && hasMore && isLoading) {
            return _buildLoadingItem();
          }

          final image = images[index];
          return _buildImageItem(context, image);
        },
      ),
    );
  }

  Widget _buildImageItem(BuildContext context, ImageModel image) {
    final theme = Theme.of(context);

    return Card(
      child: InkWell(
        onTap: () => onImageTap?.call(image),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: Stack(
          children: [
            // 图片
            Positioned.fill(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(AppTheme.radiusM),
                child: CachedNetworkImage(
                  imageUrl: image.imageUrl,
                  fit: BoxFit.cover,
                  placeholder: (context, url) => Container(
                    color: theme.colorScheme.surfaceContainerHighest,
                    child: const Center(child: CircularProgressIndicator()),
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

            // 类型标签
            Positioned(
              top: 4,
              right: 4,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: _getTypeColor(image.imageType),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  image.imageType.displayName,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),

            // 文件名（底部显示）
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.all(4),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.only(
                    bottomLeft: Radius.circular(AppTheme.radiusM),
                    bottomRight: Radius.circular(AppTheme.radiusM),
                  ),
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withValues(alpha: 0.6),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Text(
                  image.filename,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 8,
                    fontWeight: FontWeight.w500,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ],
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

  Color _getTypeColor(ImageType type) {
    switch (type) {
      case ImageType.foggy:
        return Colors.grey;
      case ImageType.clear:
        return Colors.blue;
      case ImageType.annotated:
        return Colors.green;
    }
  }
}
