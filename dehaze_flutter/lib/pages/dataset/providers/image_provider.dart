import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/dataset_model.dart' as g;
import '../../../providers/providers.dart';
import '../../../services/dataset_service.dart';
import '../models/dataset_model.dart';

/// 图片列表状态管理
///
/// 内部使用全局 DatasetItemService 获取数据项，再展开为本地 ImageModel 列表。
class ImageNotifier extends StateNotifier<AsyncValue<List<ImageModel>>> {
  ImageNotifier(this._service) : super(const AsyncValue.data([]));

  final DatasetItemService _service;
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
      final result = await _service.getList(g.DatasetItemQuery(
        datasetId: _currentDatasetId!,
        pageNum: _currentPage,
        pageSize: 20,
        keyword: _searchQuery.isEmpty ? null : _searchQuery,
      ));

      final images = <ImageModel>[];
      for (final item in result.list) {
        for (final img in _extractImagesFromItem(item, _currentDatasetId!)) {
          if (_selectedType != null && img.imageType != _selectedType) continue;
          images.add(img);
        }
      }

      if (refresh) {
        state = AsyncValue.data(images);
      } else {
        final currentList = state.value ?? [];
        state = AsyncValue.data([...currentList, ...images]);
      }

      _hasMore = result.list.isNotEmpty;
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

/// 从全局 DatasetItemVO 提取图片列表（清晰图 + 有雾图）
List<ImageModel> _extractImagesFromItem(g.DatasetItemVO item, int datasetId) {
  final images = <ImageModel>[];
  final createTime = item.createTime ?? '';

  if (item.clearImage != null) {
    images.add(_imageUrlToModel(item.clearImage!, datasetId, createTime));
  }
  if (item.hazyImages != null) {
    for (final img in item.hazyImages!) {
      images.add(_imageUrlToModel(img, datasetId, createTime));
    }
  }
  return images;
}

/// 将全局 ImageUrlVO 映射为本地 ImageModel
ImageModel _imageUrlToModel(g.ImageUrlVO img, int datasetId, String createdAt) {
  return ImageModel(
    id: img.id,
    datasetId: datasetId,
    filename: img.fileName ?? 'image_${img.id}',
    imageUrl: img.url,
    imageType: ImageTypeExtension.fromValue(img.type),
    createdAt: createdAt,
    width: img.width,
    height: img.height,
    fileSize: img.sizeBytes,
  );
}

/// 图片列表 Provider
final imageProvider =
    StateNotifierProvider<ImageNotifier, AsyncValue<List<ImageModel>>>((ref) {
  final service = ref.watch(datasetItemServiceProvider);
  return ImageNotifier(service);
});

/// 图片类型筛选 Provider
final imageTypeFilterProvider = StateProvider<ImageType?>((ref) => null);
