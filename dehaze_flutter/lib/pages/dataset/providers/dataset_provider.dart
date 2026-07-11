import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/providers.dart';
import '../models/dataset_model.dart';
import '../services/dataset_service.dart';

/// 数据集服务 Provider
final datasetServiceProvider = Provider<DatasetService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return DatasetService(dio);
});

/// 数据集列表状态管理
class DatasetNotifier extends StateNotifier<AsyncValue<List<DatasetModel>>> {
  DatasetNotifier(this._service) : super(const AsyncValue.loading());

  final DatasetService _service;
  String _searchQuery = '';

  Future<void> loadDatasets({bool refresh = false, String search = ''}) async {
    if (refresh || state is AsyncLoading) {
      state = const AsyncValue.loading();
    }

    try {
      _searchQuery = search;
      final datasets = await _service.getDatasets(keywords: search.isEmpty ? null : search);
      state = AsyncValue.data(datasets);
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
    StateNotifierProvider<DatasetNotifier, AsyncValue<List<DatasetModel>>>((ref) {
  final service = ref.watch(datasetServiceProvider);
  return DatasetNotifier(service);
});

/// 选中的数据集 Provider
final selectedDatasetProvider = StateProvider<DatasetModel?>((ref) => null);
