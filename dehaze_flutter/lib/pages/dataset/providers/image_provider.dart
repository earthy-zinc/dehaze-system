import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/dataset_model.dart';
import '../providers/dataset_provider.dart';
import '../services/dataset_service.dart';

/// 图片列表状态管理
class ImageNotifier extends StateNotifier<AsyncValue<List<ImageModel>>> {
  ImageNotifier(this._service) : super(const AsyncValue.data([]));

  final DatasetService _service;
  int _currentPage = 1;
  bool _hasMore = true;
  ImageType? _selectedType;
  String _searchQuery = '';
  int? _currentDatasetId;

  Future<void> loadImages({
    required int datasetId,
    bool refresh = false,
    ImageType? imageType,
    String search = '',
  }) async {
    if (refresh) {
      _currentPage = 1;
      _hasMore = true;
      _selectedType = imageType;
      _searchQuery = search;
      _currentDatasetId = datasetId;
      state = const AsyncValue.loading();
    }

    if (!_hasMore || _currentDatasetId == null) return;

    try {
      final response = await _service.getDatasetImages(
        datasetId: _currentDatasetId!,
        pageNum: _currentPage,
        imageType: _selectedType,
        keywords: _searchQuery.isEmpty ? null : _searchQuery,
      );

      if (refresh) {
        state = AsyncValue.data(response.list);
      } else {
        final currentList = state.value ?? [];
        state = AsyncValue.data([...currentList, ...response.list]);
      }

      _hasMore = response.list.isNotEmpty;
      _currentPage++;
    } catch (e, stackTrace) {
      state = AsyncValue.error(e, stackTrace);
    }
  }

  Future<void> filterByType(ImageType? type) async {
    if (type != _selectedType && _currentDatasetId != null) {
      await loadImages(
        datasetId: _currentDatasetId!,
        refresh: true,
        imageType: type,
      );
    }
  }

  Future<void> searchImages(String query) async {
    if (query != _searchQuery && _currentDatasetId != null) {
      await loadImages(
        datasetId: _currentDatasetId!,
        refresh: true,
        search: query,
      );
    }
  }

  Future<void> refresh() async {
    if (_currentDatasetId != null) {
      await loadImages(datasetId: _currentDatasetId!, refresh: true);
    }
  }

  Future<void> loadMore() async {
    if (_hasMore && _currentDatasetId != null && state is AsyncData) {
      await loadImages(datasetId: _currentDatasetId!, refresh: false);
    }
  }
}

/// 图片列表 Provider
final imageProvider =
    StateNotifierProvider<ImageNotifier, AsyncValue<List<ImageModel>>>((ref) {
  final service = ref.watch<DatasetService>(datasetServiceProvider);
  return ImageNotifier(service);
});

/// 图片类型筛选 Provider
final imageTypeFilterProvider = StateProvider<ImageType?>((ref) => null);
