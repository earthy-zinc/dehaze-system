import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../utils/responsive_utils.dart';
import 'models/image_input_model.dart';
import 'providers/image_input_provider.dart';
import 'providers/sample_provider.dart';
import 'widgets/camera_capture.dart';
import 'widgets/history_list.dart';
import 'widgets/image_preview.dart';
import 'widgets/input_method_selector.dart';
import 'widgets/sample_gallery.dart';
import 'widgets/upload_area.dart';

/// 图像输入页面
///
/// 与设计稿 imageInput.js 功能对应
/// 支持：上传图片、拍照、样例图片、历史记录
class ImageInputPage extends ConsumerStatefulWidget {
  const ImageInputPage({super.key});

  @override
  ConsumerState<ImageInputPage> createState() => _ImageInputPageState();
}

class _ImageInputPageState extends ConsumerState<ImageInputPage> {
  @override
  Widget build(BuildContext context) {
    final selectedImage = ref.watch(selectedImageProvider);
    final currentMethod = ref.watch(inputMethodProvider);
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1200,
        padding: EdgeInsets.zero,
        child: CustomScrollView(
          slivers: [
            // 页面头部
            _buildHeaderSliver(theme),

            // 主内容区域
            SliverPadding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              sliver: SliverToBoxAdapter(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // 图像输入卡片
                    _buildInputCard(theme, currentMethod),

                    const SizedBox(height: 24),

                    // 图片预览（如果有选中的图片）
                    if (selectedImage != null)
                      ImagePreview(
                        onNextStep: _goToAlgorithmSelect,
                      ),

                    // 快速体验卡片（如果没有选中图片）
                    if (selectedImage == null) ...[
                      const SizedBox(height: 16),
                      _buildQuickStartCard(theme),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 构建页面头部
  Widget _buildHeaderSliver(ThemeData theme) => SliverToBoxAdapter(
        child: Container(
          padding: ResponsiveUtils.getResponsivePadding(context),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            border: Border(
              bottom: BorderSide(
                color: theme.dividerColor,
                width: 1,
              ),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 标题行
              Row(
                children: [
                  Icon(
                    Icons.image_outlined,
                    color: const Color(0xFF3B82F6),
                    size: 24,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '图像输入',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '选择图片开始去雾处理',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );

  /// 构建图像输入卡片
  Widget _buildInputCard(ThemeData theme, InputMethod currentMethod) => Container(
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.05),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 输入方式选择器
            Padding(
              padding: const EdgeInsets.all(16),
              child: InputMethodSelector(
                onMethodChanged: (_) {
                  // 切换输入方式时清除选中的图片
                  ref.read(imageInputProvider.notifier).clearSelection();
                },
              ),
            ),

            const Divider(height: 1),

            // 内容区域
            _buildContentArea(theme, currentMethod),
          ],
        ),
      );

  /// 构建内容区域
  Widget _buildContentArea(ThemeData theme, InputMethod currentMethod) {
    // 根据不同输入方式显示不同内容
    switch (currentMethod) {
      case InputMethod.upload:
        return const Padding(
          padding: EdgeInsets.all(16),
          child: UploadArea(),
        );
      case InputMethod.camera:
        return const Padding(
          padding: EdgeInsets.all(16),
          child: CameraCapture(),
        );
      case InputMethod.sample:
        return SizedBox(
          height: 400,
          child: const Padding(
            padding: EdgeInsets.only(top: 16, bottom: 16),
            child: SampleGallery(),
          ),
        );
      case InputMethod.history:
        return SizedBox(
          height: 400,
          child: const Padding(
            padding: EdgeInsets.only(top: 16, bottom: 16),
            child: HistoryList(),
          ),
        );
    }
  }

  /// 构建快速体验卡片
  Widget _buildQuickStartCard(ThemeData theme) => Container(
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF3B82F6), // blue-500
              Color(0xFF6366F1), // indigo-500
            ],
          ),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFF3B82F6).withValues(alpha: 0.3),
              blurRadius: 16,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(
                          Icons.bolt,
                          color: Colors.white,
                          size: 24,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          '快速体验',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '使用样例图片快速体验去雾效果',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: Colors.white.withValues(alpha: 0.9),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              FilledButton(
                onPressed: _quickStart,
                style: FilledButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: const Color(0xFF3B82F6),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text('立即体验'),
              ),
            ],
          ),
        ),
      );

  /// 快速体验 - 随机选择样例图片
  void _quickStart() {
    // 切换到样例图片模式
    ref.read(inputMethodProvider.notifier).state = InputMethod.sample;

    // 加载样例并选择随机图片
    final sampleNotifier = ref.read(sampleProvider.notifier);
    sampleNotifier.loadSamples(refresh: true).then((_) {
      final randomSample = sampleNotifier.getRandomSample();
      if (randomSample != null) {
        ref.read(imageInputProvider.notifier).selectSampleImage(randomSample);
      }
    });
  }

  /// 跳转到算法选择页面
  void _goToAlgorithmSelect() {
    // TODO: 实现跳转到算法选择页面
    // context.go('/algorithm-select');

    // 临时提示
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('图片已选择，即将跳转到算法选择页面'),
        duration: Duration(seconds: 2),
      ),
    );
  }
}
