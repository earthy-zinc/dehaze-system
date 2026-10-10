import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/dataset_model.dart';
import '../../../providers/providers.dart';
import '../../../services/dataset_service.dart';

/// 数据集列表状态管理
class DatasetNotifier extends StateNotifier<AsyncValue<List<Dataset>>> {
  DatasetNotifier(this._service) : super(const AsyncValue.loading());

  /// 数据集选择页一次性加载上限（页面无分页加载，保持有界请求）
  static const int _maxDatasets = 100;

  final DatasetService _service;
  String _searchQuery = '';

  Future<void> loadDatasets({bool refresh = false, String search = ''}) async {
    if (refresh || state is AsyncLoading) {
      state = const AsyncValue.loading();
    }

    try {
      _searchQuery = search;
      final result = await _service.getList(
        DatasetQuery(
          keyword: search.isEmpty ? null : search,
          pageSize: _maxDatasets,
        ),
      );
      state = AsyncValue.data(result.list);
    } catch (e, stackTrace) {
      state = AsyncValue.error(e, stackTrace);
    }
  }

  Future<void> searchDatasets(String query) async {
    if (query != _searchQuery) {
      await loadDatasets(refresh: true, search: query);
    }
  }

  Future<void> refresh() async {
    await loadDatasets(refresh: true, search: _searchQuery);
  }
}

/// 数据集列表 Provider
final datasetProvider =
    StateNotifierProvider<DatasetNotifier, AsyncValue<List<Dataset>>>((ref) {
      final service = ref.watch(datasetServiceProvider);
      return DatasetNotifier(service);
    });

/// 选中的数据集 Provider
final selectedDatasetProvider = StateProvider<Dataset?>((ref) => null);
