import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/image_input_model.dart';
import 'image_input_provider.dart';

// ==================== 状态 Provider ====================

/// 样例分类筛选
final sampleCategoryProvider = StateProvider<SampleCategory>((ref) => SampleCategory.all);

// ==================== 样例图片 Notifier ====================

/// 样例图片状态管理
class SampleNotifier extends StateNotifier<AsyncValue<List<SampleImageModel>>> {
  SampleNotifier(this._ref) : super(const AsyncValue.loading());

  final Ref _ref;

  /// 加载样例图片
  Future<void> loadSamples({SampleCategory? category, bool refresh = false}) async {
    if (!refresh && state is AsyncData) {
      // 如果已有数据且不需要刷新，只进行筛选
      if (category != null) {
        await filterByCategory(category);
      }
      return;
    }

    state = const AsyncValue.loading();

    try {
      final service = _ref.read(imageInputServiceProvider);
      final samples = await service.fetchSamples(category: category);
      state = AsyncValue.data(samples);
    } catch (e, stackTrace) {
      state = AsyncValue.error(e, stackTrace);
    }
  }

  /// 按分类筛选
  Future<void> filterByCategory(SampleCategory category) async {
    _ref.read(sampleCategoryProvider.notifier).state = category;

    state = const AsyncValue.loading();

    try {
      final service = _ref.read(imageInputServiceProvider);
      final samples = await service.fetchSamples(category: category);
      state = AsyncValue.data(samples);
    } catch (e, stackTrace) {
      state = AsyncValue.error(e, stackTrace);
    }
  }

  /// 刷新
  Future<void> refresh() async {
    final category = _ref.read(sampleCategoryProvider);
    await loadSamples(category: category, refresh: true);
  }

  /// 获取随机样例图片（用于快速体验）
  SampleImageModel? getRandomSample() {
    final samples = state.value;
    if (samples == null || samples.isEmpty) {
      return null;
    }

    // 随机选择一张
    final randomIndex = DateTime.now().millisecondsSinceEpoch % samples.length;
    return samples[randomIndex];
  }
}

/// 样例图片 Provider
final sampleProvider =
    StateNotifierProvider<SampleNotifier, AsyncValue<List<SampleImageModel>>>((ref) {
  return SampleNotifier(ref);
});

/// 按分类分组的样例图片
final groupedSamplesProvider = Provider<Map<SampleCategory, List<SampleImageModel>>>((ref) {
  final samplesAsync = ref.watch(sampleProvider);

  return samplesAsync.when(
    data: (samples) {
      final grouped = <SampleCategory, List<SampleImageModel>>{};

      for (final sample in samples) {
        grouped.putIfAbsent(sample.category, () => []).add(sample);
      }

      return grouped;
    },
    loading: () => {},
    error: (_, _) => {},
  );
});

/// 各分类的数量统计
final sampleCountsProvider = Provider<Map<SampleCategory, int>>((ref) {
  final grouped = ref.watch(groupedSamplesProvider);

  final counts = <SampleCategory, int>{
    SampleCategory.all: 0,
  };

  for (final category in SampleCategory.values) {
    if (category != SampleCategory.all) {
      counts[category] = grouped[category]?.length ?? 0;
      counts[SampleCategory.all] = counts[SampleCategory.all]! + counts[category]!;
    }
  }

  return counts;
});
