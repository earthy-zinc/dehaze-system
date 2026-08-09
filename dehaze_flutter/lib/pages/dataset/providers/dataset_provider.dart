import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/dataset_model.dart' as g;
import '../../../providers/providers.dart';
import '../../../services/dataset_service.dart';

/// 数据集列表状态管理
class DatasetNotifier extends StateNotifier<AsyncValue<List<g.Dataset>>> {
  DatasetNotifier(this._service) : super(const AsyncValue.loading());

  final DatasetService _service;
  String _searchQuery = '';

  Future<void> loadDatasets({bool refresh = false, String search = ''}) async {
    if (refresh || state is AsyncLoading) {
      state = const AsyncValue.loading();
    }

    try {
      _searchQuery = search;
      final result = await _service.getList(g.DatasetQuery(
        keyword: search.isEmpty ? null : search,
        pageSize: 100,
      ));
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
    StateNotifierProvider<DatasetNotifier, AsyncValue<List<g.Dataset>>>((ref) {
  final service = ref.watch(datasetServiceProvider);
  return DatasetNotifier(service);
});

/// 选中的数据集 Provider
final selectedDatasetProvider = StateProvider<g.Dataset?>((ref) => null);
