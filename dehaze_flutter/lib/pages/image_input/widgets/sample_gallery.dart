import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../utils/responsive_utils.dart';
import '../models/image_input_model.dart';
import '../providers/image_input_provider.dart';
import '../providers/sample_provider.dart';
import 'sample_category_tabs.dart';
import 'sample_image_card.dart';

/// 样例图片库
///
/// 包含分类标签和图片网格
class SampleGallery extends ConsumerStatefulWidget {
  const SampleGallery({super.key});

  @override
  ConsumerState<SampleGallery> createState() => _SampleGalleryState();
}

class _SampleGalleryState extends ConsumerState<SampleGallery> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(sampleProvider.notifier).loadSamples(refresh: true);
    });
  }

  @override
  Widget build(BuildContext context) {
    final samplesAsync = ref.watch(sampleProvider);
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 分类标签
        const SampleCategoryTabs(),

        const SizedBox(height: 16),

        // 图片网格
        Expanded(
          child: samplesAsync.when(
            data: (samples) {
              if (samples.isEmpty) {
                return _buildEmptyState(theme);
              }
              return _buildGrid(samples);
            },
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (error, stack) => _buildErrorState(theme, error.toString()),
          ),
        ),
      ],
    );
  }

  Widget _buildGrid(List<SampleImageModel> samples) {
    final crossAxisCount = ResponsiveUtils.getGridCrossAxisCount(
      context,
      mobile: 2,
      tablet: 3,
      desktop: 4,
      largeDesktop: 5,
    );

    return GridView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 0.85,
      ),
      itemCount: samples.length,
      itemBuilder: (context, index) {
        final sample = samples[index];
        return SampleImageCard(
          sample: sample,
          onTap: () => _selectSample(sample),
        );
      },
    );
  }

  Widget _buildEmptyState(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.collections_outlined,
              size: 64,
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.5),
            ),
            const SizedBox(height: 16),
            Text(
              '暂无样例图片',
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );

  Widget _buildErrorState(ThemeData theme, String error) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: theme.colorScheme.error,
            ),
            const SizedBox(height: 16),
            Text(
              '加载失败',
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              error,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: () => ref.read(sampleProvider.notifier).refresh(),
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );

  void _selectSample(SampleImageModel sample) {
    ref.read(imageInputProvider.notifier).selectSampleImage(sample);
  }
}
