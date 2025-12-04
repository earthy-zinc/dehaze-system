import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../../theme/app_theme.dart';
import 'models/dataset_model.dart';
import 'providers/dataset_provider.dart';
import 'providers/image_provider.dart';
import 'widgets/dataset_card.dart';
import 'widgets/dataset_info_card.dart';
import 'widgets/image_grid.dart';
import 'widgets/type_filter_tabs.dart';

class DatasetPage extends ConsumerStatefulWidget {
  const DatasetPage({super.key});

  @override
  ConsumerState<DatasetPage> createState() => _DatasetPageState();
}

class _DatasetPageState extends ConsumerState<DatasetPage> {
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // 加载数据集
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(datasetProvider.notifier).loadDatasets(refresh: true);
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final datasetsAsync = ref.watch(datasetProvider);
    final selectedDataset = ref.watch(selectedDatasetProvider);

    return Scaffold(
      body: Column(
        children: [
          // 搜索栏
          _buildSearchBar(),

          // 内容区域
          Expanded(
            child: selectedDataset != null
                ? _buildDetailView()
                : _buildListView(datasetsAsync),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchBar() {
    final selectedDataset = ref.watch(selectedDatasetProvider);

    return Container(
      padding: EdgeInsets.all(AppTheme.spacingM),
      child: TextField(
        controller: _searchController,
        decoration: InputDecoration(
          hintText: selectedDataset != null ? '搜索图片...' : '搜索数据集...',
          prefixIcon: const Icon(Icons.search),
          suffixIcon: _searchController.text.isNotEmpty
              ? IconButton(
                  onPressed: () {
                    _searchController.clear();
                    _performSearch('');
                  },
                  icon: const Icon(Icons.clear),
                )
              : null,
        ),
        onChanged: _performSearch,
      ),
    );
  }

  Widget _buildListView(AsyncValue<List<DatasetModel>> datasetsAsync) =>
      RefreshIndicator(
        onRefresh: () async {
          await ref.read(datasetProvider.notifier).refresh();
        },
        child: datasetsAsync.when(
          data: (datasets) {
            if (datasets.isEmpty) {
              return _buildEmptyState('暂无数据集');
            }

            return ListView.builder(
              padding: EdgeInsets.symmetric(horizontal: AppTheme.spacingM),
              itemCount: datasets.length,
              itemBuilder: (context, index) {
                final dataset = datasets[index];
                return Padding(
                  padding: EdgeInsets.only(bottom: AppTheme.spacingM),
                  child: DatasetCard(
                    dataset: dataset,
                    onTap: () => _showDatasetDetail(dataset),
                  ),
                );
              },
            );
          },
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (error, stack) => _buildErrorState(error.toString()),
        ),
      );

  Widget _buildDetailView() {
    final dataset = ref.watch(selectedDatasetProvider)!;
    final imagesAsync = ref.watch(imageProvider);

    return Column(
      children: [
        // 返回按钮和数据集信息
        Container(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: Column(
            children: [
              Row(
                children: [
                  IconButton(
                    onPressed: _backToList,
                    icon: const Icon(Icons.arrow_back),
                  ),
                  SizedBox(width: AppTheme.spacingS),
                  Expanded(
                    child: Text(
                      '数据集详情',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                  ),
                ],
              ),
              SizedBox(height: AppTheme.spacingM),
              DatasetInfoCard(dataset: dataset),
            ],
          ),
        ),

        // 类型筛选标签
        TypeFilterTabs(
          selectedType: ref.watch(imageTypeFilterProvider),
          totalCount: dataset.totalImages,
          foggyCount: dataset.foggyCount,
          clearCount: dataset.clearCount,
          annotatedCount: dataset.annotatedCount,
          onTypeChanged: (type) {
            ref.read(imageTypeFilterProvider.notifier).state = type;
            ref.read(imageProvider.notifier).filterByType(type);
          },
        ),

        // 图片网格
        Expanded(
          child: imagesAsync.when(
            data: (images) =>
                ImageGrid(images: images, onImageTap: _showImageViewer),
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (error, stack) => _buildErrorState(error.toString()),
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState(String message) => Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.storage_outlined,
          size: 64,
          color: Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.3),
        ),
        SizedBox(height: AppTheme.spacingM),
        Text(
          message,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
            color: Theme.of(
              context,
            ).colorScheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
      ],
    ),
  );

  Widget _buildErrorState(String error) => Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(
          Icons.error_outline,
          size: 64,
          color: Theme.of(context).colorScheme.error,
        ),
        SizedBox(height: AppTheme.spacingM),
        Text('加载失败', style: Theme.of(context).textTheme.titleLarge),
        SizedBox(height: AppTheme.spacingS),
        Text(
          error,
          style: Theme.of(context).textTheme.bodyMedium,
          textAlign: TextAlign.center,
        ),
        SizedBox(height: AppTheme.spacingM),
        ElevatedButton(
          onPressed: () {
            final selectedDataset = ref.read(selectedDatasetProvider);
            if (selectedDataset != null) {
              ref.read(imageProvider.notifier).refresh();
            } else {
              ref.read(datasetProvider.notifier).refresh();
            }
          },
          child: const Text('重试'),
        ),
      ],
    ),
  );

  void _performSearch(String query) {
    final selectedDataset = ref.read(selectedDatasetProvider);

    if (selectedDataset != null) {
      // 搜索图片
      ref.read(imageProvider.notifier).searchImages(query);
    } else {
      // 搜索数据集
      ref.read(datasetProvider.notifier).searchDatasets(query);
    }
  }

  void _showDatasetDetail(DatasetModel dataset) {
    ref.read(selectedDatasetProvider.notifier).state = dataset;
    // 加载该数据集的图片
    ref
        .read(imageProvider.notifier)
        .loadImages(datasetId: dataset.id, refresh: true);
    _searchController.clear();
  }

  void _backToList() {
    ref.read(selectedDatasetProvider.notifier).state = null;
    ref.read(imageTypeFilterProvider.notifier).state = null;
    _searchController.clear();
  }

  void _showImageViewer(ImageModel image) {
    showDialog<void>(
      context: context,
      builder: (context) => Dialog(
        child: Stack(
          children: [
            Image.network(image.imageUrl),
            Positioned(
              top: 16,
              right: 16,
              child: IconButton(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close),
                style: IconButton.styleFrom(
                  backgroundColor: Colors.black.withValues(alpha: 0.5),
                  foregroundColor: Colors.white,
                ),
              ),
            ),
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: EdgeInsets.all(AppTheme.spacingM),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      Colors.black.withValues(alpha: 0.8),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      image.filename,
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    SizedBox(height: AppTheme.spacingS),
                    Text(
                      '${image.imageType.displayName} • ${image.width} × ${image.height}',
                      style: TextStyle(color: Colors.white, fontSize: 14),
                    ),
                    if (image.description != null) ...[
                      SizedBox(height: AppTheme.spacingS),
                      Text(
                        image.description!,
                        style: TextStyle(color: Colors.white, fontSize: 12),
                      ),
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
}
