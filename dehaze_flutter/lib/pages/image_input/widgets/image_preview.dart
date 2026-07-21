import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../widgets/dehaze_image.dart';
import '../models/image_input_model.dart';
import '../providers/image_input_provider.dart';

/// 图片预览组件
///
/// 显示选中图片的预览和信息
/// 提供移除和下一步操作
class ImagePreview extends ConsumerWidget {
  const ImagePreview({super.key, this.onNextStep});

  final VoidCallback? onNextStep;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedImage = ref.watch(selectedImageProvider);
    final theme = Theme.of(context);

    if (selectedImage == null) {
      return const SizedBox.shrink();
    }

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 头部
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Icon(
                  Icons.image_outlined,
                  color: theme.colorScheme.primary,
                  size: 20,
                ),
                const SizedBox(width: 8),
                Text(
                  '图片预览',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                IconButton(
                  onPressed: () => ref.read(imageInputProvider.notifier).clearSelection(),
                  icon: const Icon(Icons.close),
                  iconSize: 20,
                  tooltip: '移除',
                  style: IconButton.styleFrom(
                    foregroundColor: theme.colorScheme.error,
                  ),
                ),
              ],
            ),
          ),

          // 图片预览区域
          Container(
            height: 280,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            decoration: BoxDecoration(
              color: theme.colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(12),
            ),
            clipBehavior: Clip.antiAlias,
            child: _buildImage(selectedImage),
          ),

          // 图片信息
          Padding(
            padding: const EdgeInsets.all(16),
            child: _buildImageInfo(context, selectedImage),
          ),

          // 下一步按钮
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: FilledButton.icon(
              onPressed: onNextStep,
              icon: const Icon(Icons.arrow_forward),
              label: const Text('下一步：选择算法'),
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImage(SelectedImageModel image) {
    // 跨平台渲染：字节流优先（本地选择/拍摄），网络地址次之（样例图片）
    return DehazeImage(
      bytes: image.bytes,
      url: image.url,
      fit: BoxFit.contain,
    );
  }

  Widget _buildImageInfo(BuildContext context, SelectedImageModel image) {
    final theme = Theme.of(context);

    return Row(
      children: [
        // 文件名
        Expanded(
          child: Row(
            children: [
              Icon(
                Icons.insert_drive_file_outlined,
                size: 16,
                color: theme.colorScheme.onSurfaceVariant,
              ),
              const SizedBox(width: 4),
              Expanded(
                child: Text(
                  image.filename,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(width: 16),

        // 文件大小
        Row(
          children: [
            Icon(
              Icons.data_usage_outlined,
              size: 16,
              color: const Color(0xFF3B82F6),
            ),
            const SizedBox(width: 4),
            Text(
              _formatFileSize(image.fileSize),
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),

        const SizedBox(width: 16),

        // 尺寸
        if (image.width > 0 && image.height > 0)
          Row(
            children: [
              Icon(
                Icons.aspect_ratio_outlined,
                size: 16,
                color: const Color(0xFF10B981),
              ),
              const SizedBox(width: 4),
              Text(
                '${image.width} × ${image.height}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
      ],
    );
  }

  String _formatFileSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(2)} MB';
  }
}
