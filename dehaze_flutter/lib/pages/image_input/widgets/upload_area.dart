import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/image_input_model.dart';
import '../providers/image_input_provider.dart';

/// 图片上传区域
///
/// 支持点击上传
/// 显示上传进度
/// 自动压缩大图片
class UploadArea extends ConsumerWidget {
  const UploadArea({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final uploadProgress = ref.watch(uploadProgressProvider);
    final theme = Theme.of(context);

    return GestureDetector(
      onTap: uploadProgress.status == UploadStatus.idle ||
              uploadProgress.status == UploadStatus.error ||
              uploadProgress.status == UploadStatus.success
          ? () => ref.read(imageInputProvider.notifier).pickImage()
          : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        padding: const EdgeInsets.all(32),
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: _getBorderColor(uploadProgress.status, theme),
            width: 2,
            strokeAlign: BorderSide.strokeAlignInside,
          ),
        ),
        child: _buildContent(context, ref, uploadProgress),
      ),
    );
  }

  Color _getBorderColor(UploadStatus status, ThemeData theme) {
    switch (status) {
      case UploadStatus.idle:
      case UploadStatus.selecting:
        return theme.dividerColor;
      case UploadStatus.validating:
      case UploadStatus.compressing:
      case UploadStatus.uploading:
        return const Color(0xFF3B82F6); // blue-500
      case UploadStatus.success:
        return const Color(0xFF10B981); // emerald-500
      case UploadStatus.error:
        return const Color(0xFFEF4444); // red-500
    }
  }

  Widget _buildContent(BuildContext context, WidgetRef ref, UploadProgress progress) {
    final theme = Theme.of(context);

    switch (progress.status) {
      case UploadStatus.idle:
        return _buildIdleState(theme);
      case UploadStatus.selecting:
        return _buildLoadingState(theme, '选择图片中...');
      case UploadStatus.validating:
        return _buildProgressState(theme, progress, '验证图片...');
      case UploadStatus.compressing:
        return _buildProgressState(theme, progress, '压缩图片中...');
      case UploadStatus.uploading:
        return _buildProgressState(theme, progress, '处理中...');
      case UploadStatus.success:
        return _buildSuccessState(theme);
      case UploadStatus.error:
        return _buildErrorState(theme, progress.errorMessage);
    }
  }

  Widget _buildIdleState(ThemeData theme) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFF3B82F6).withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.cloud_upload_outlined,
              size: 48,
              color: Color(0xFF3B82F6),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            '点击上传图片',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: theme.colorScheme.onSurface,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '支持 JPG、PNG、WEBP、HEIC 格式',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            '最大 20MB，建议分辨率 ≥640×480',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      );

  Widget _buildLoadingState(ThemeData theme, String message) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 48,
            height: 48,
            child: CircularProgressIndicator(strokeWidth: 3),
          ),
          const SizedBox(height: 16),
          Text(
            message,
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface,
            ),
          ),
        ],
      );

  Widget _buildProgressState(ThemeData theme, UploadProgress progress, String message) =>
      Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 64,
            height: 64,
            child: Stack(
              alignment: Alignment.center,
              children: [
                CircularProgressIndicator(
                  value: progress.progress,
                  strokeWidth: 4,
                  backgroundColor: theme.dividerColor,
                ),
                Text(
                  '${(progress.progress * 100).toInt()}%',
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text(
            message,
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface,
            ),
          ),
        ],
      );

  Widget _buildSuccessState(ThemeData theme) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: const BoxDecoration(
              color: Color(0xFF10B981),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.check,
              size: 32,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            '上传成功',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: const Color(0xFF10B981),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '点击重新选择',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      );

  Widget _buildErrorState(ThemeData theme, String? errorMessage) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFEF4444).withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.error_outline,
              size: 32,
              color: Color(0xFFEF4444),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            errorMessage ?? '上传失败',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: const Color(0xFFEF4444),
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            '点击重试',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      );
}
