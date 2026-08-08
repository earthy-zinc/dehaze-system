import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/dataset_model.dart' as g;
import '../../../providers/providers.dart';
import '../../../services/dataset_service.dart';
import '../models/dataset_model.dart';

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
      final result = await _service.getList(g.DatasetQuery(
        keyword: search.isEmpty ? null : search,
        pageSize: 100,
      ));
      state = AsyncValue.data(
        result.list.map(_datasetToModel).toList(),
      );
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

/// 将全局 Dataset 映射为本地 DatasetModel
DatasetModel _datasetToModel(g.Dataset d) {
  return DatasetModel(
    id: d.id,
    parentId: d.parentId,
    name: d.name,
    type: d.type,
    path: d.path,
    description: d.description,
    usageCount: null,
    createBy: null,
    createTime: d.createTime ?? '',
    updateTime: d.updateTime,
    updateBy: null,
    children: d.children?.map(_datasetToModel).toList() ?? const [],
    hasChildren: d.hasChildren ?? false,
    total: d.total,
    status: d.status,
  );
}

/// 数据集列表 Provider
final datasetProvider =
    StateNotifierProvider<DatasetNotifier, AsyncValue<List<DatasetModel>>>((ref) {
  final service = ref.watch(datasetServiceProvider);
  return DatasetNotifier(service);
});

/// 选中的数据集 Provider
final selectedDatasetProvider = StateProvider<DatasetModel?>((ref) => null);
