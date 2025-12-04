import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../providers/providers.dart';
import '../models/dataset_model.dart';
import '../services/dataset_service.dart';

// 数据集服务Provider
final datasetServiceProvider = Provider<DatasetService>((ref) {
  final dio = ref.watch<Dio>(dioClientProvider);
  return DatasetService(dio);
});

// 数据集列表Provider
class DatasetNotifier extends StateNotifier<AsyncValue<List<DatasetModel>>> {
  DatasetNotifier(this._service) : super(const AsyncValue.loading());

  final DatasetService _service;
  int _currentPage = 1;
  bool _hasMore = true;
  String _searchQuery = '';

  Future<void> loadDatasets({bool refresh = false, String search = ''}) async {
    if (refresh) {
      _currentPage = 1;
      _hasMore = true;
      _searchQuery = search;
      state = const AsyncValue.loading();
    }

    if (!_hasMore) {
      return;
    }

    state = const AsyncValue.loading();

    try {
      final response = await _service.fetchDatasets(
        page: _currentPage,
        search: _searchQuery,
      );

      if (refresh) {
        state = AsyncValue.data(response.list);
      } else {
        final currentList = state.value ?? [];
        state = AsyncValue.data([...currentList, ...response.list]);
      }

      _currentPage++;
      _hasMore = response.page < response.totalPages;
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

  Future<void> loadMore() async {
    if (_hasMore && state is AsyncData) {
      await loadDatasets(refresh: false);
    }
  }
}

final datasetProvider =
    StateNotifierProvider<DatasetNotifier, AsyncValue<List<DatasetModel>>>((
      ref,
    ) {
      final service = ref.watch(datasetServiceProvider);
      return DatasetNotifier(service);
    });

// 选中的数据集Provider
final selectedDatasetProvider = StateProvider<DatasetModel?>((ref) => null);
