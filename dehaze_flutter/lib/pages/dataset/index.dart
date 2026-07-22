import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../utils/responsive_utils.dart';
import 'models/dataset_model.dart';
import 'providers/dataset_provider.dart';
import 'providers/image_provider.dart';
import 'widgets/dataset_card.dart';
import 'widgets/dataset_info_card.dart';
import 'widgets/image_grid.dart';
import 'widgets/type_filter_tabs.dart';

/// 数据集管理页面
///
/// 与设计稿 dataset.js 功能对应
/// 支持：数据集列表、数据集详情、图片浏览、类型筛选、搜索
class DatasetPage extends ConsumerStatefulWidget {
  const DatasetPage({super.key, this.initialDatasetId});

  /// 初始数据集ID，用于直接跳转到数据集详情
  final int? initialDatasetId;

  @override
  ConsumerState<DatasetPage> createState() => _DatasetPageState();
}

class _DatasetPageState extends ConsumerState<DatasetPage> {
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(datasetProvider.notifier).loadDatasets(refresh: true);

      if (widget.initialDatasetId != null) {
        _loadDatasetById(widget.initialDatasetId!);
      }
    });
  }

  Future<void> _loadDatasetById(int datasetId) async {
    final datasets = ref.read(datasetProvider).value ?? [];
    final dataset = datasets.where((d) => d.id == datasetId).firstOrNull;
    if (dataset != null) {
      _showDatasetDetail(dataset);
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    _searchFocusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final datasetsAsync = ref.watch(datasetProvider);
    final selectedDataset = ref.watch(selectedDatasetProvider);
    final isWide = ResponsiveUtils.isWideScreen(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1400,
        padding: EdgeInsets.zero,
        child: selectedDataset != null
            ? _buildDetailView(selectedDataset, isWide)
            : _buildListView(datasetsAsync, isWide),
      ),
    );
  }

  /// 构建页面头部（作为 Sliver）
  Widget _buildHeaderSliver(DatasetModel? selectedDataset) {
    final theme = Theme.of(context);

    return SliverToBoxAdapter(
      child: Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(
              color: theme.dividerColor,
              width: 1,
            ),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 标题行
            Row(
              children: [
                if (selectedDataset != null) ...[
                  IconButton(
                    onPressed: _backToList,
                    icon: const Icon(Icons.arrow_back),
                    tooltip: '返回列表',
                  ),
                  const SizedBox(width: 8),
                ],
                Icon(
                  Icons.storage_outlined,
                  color: const Color(0xFF14B8A6),
                  size: 24,
                ),
                const SizedBox(width: 8),
                Text(
                  selectedDataset != null ? '数据集详情' : '数据集管理',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 8),

            // 描述文字
            Text(
              selectedDataset != null
                  ? '浏览 ${selectedDataset.name} 中的图片'
                  : '浏览和管理图像去雾数据集',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),

            const SizedBox(height: 16),

            // 搜索栏
            _buildSearchBar(selectedDataset),
          ],
        ),
      ),
    );
  }

  /// 构建搜索栏
  Widget _buildSearchBar(DatasetModel? selectedDataset) => TextField(
        controller: _searchController,
        focusNode: _searchFocusNode,
        decoration: InputDecoration(
          hintText: selectedDataset != null ? '搜索图片...' : '搜索数据集或图片...',
          prefixIcon: const Icon(Icons.search),
          suffixIcon: _searchController.text.isNotEmpty
              ? IconButton(
                  onPressed: () {
                    _searchController.clear();
                    _performSearch('');
                    _searchFocusNode.unfocus();
                  },
                  icon: const Icon(Icons.clear),
                  tooltip: '清除',
                )
              : null,
          filled: true,
          fillColor: Theme.of(context).colorScheme.surfaceContainerHighest,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(
              color: Color(0xFF14B8A6),
              width: 2,
            ),
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 12,
          ),
        ),
        onChanged: _performSearch,
      );

  /// 构建数据集列表视图
  Widget _buildListView(
    AsyncValue<List<DatasetModel>> datasetsAsync,
    bool isWide,
  ) =>
      RefreshIndicator(
        onRefresh: () async {
          await ref.read(datasetProvider.notifier).refresh();
        },
        child: datasetsAsync.when(
          data: (datasets) {
            if (datasets.isEmpty) {
              return CustomScrollView(
                slivers: [
                  _buildHeaderSliver(null),
                  SliverFillRemaining(
                    child: _buildEmptyState('暂无数据集', Icons.folder_open_outlined),
                  ),
                ],
              );
            }

            // 使用 CustomScrollView 让 Header 跟随滚动
            return CustomScrollView(
              slivers: [
                _buildHeaderSliver(null),
                // 响应式布局：宽屏使用网格，窄屏使用列表
                if (isWide)
                  _buildDatasetGridSliver(datasets)
                else
                  _buildDatasetListSliver(datasets),
              ],
            );
          },
          loading: () => CustomScrollView(
            slivers: [
              _buildHeaderSliver(null),
              const SliverFillRemaining(
                child: Center(child: CircularProgressIndicator()),
              ),
            ],
          ),
          error: (error, stack) => CustomScrollView(
            slivers: [
              _buildHeaderSliver(null),
              SliverFillRemaining(
                child: _buildErrorState(error.toString()),
              ),
            ],
          ),
        ),
      );

  /// 构建数据集网格（宽屏）- Sliver 版本
  Widget _buildDatasetGridSliver(List<DatasetModel> datasets) {
    final crossAxisCount = ResponsiveUtils.getGridCrossAxisCount(
      context,
      mobile: 1,
      tablet: 2,
      desktop: 2,
      largeDesktop: 3,
    );

    return SliverPadding(
      padding: ResponsiveUtils.getResponsivePadding(context),
      sliver: SliverGrid(
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: crossAxisCount,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
          childAspectRatio: 2.5,
        ),
        delegate: SliverChildBuilderDelegate(
          (context, index) => DatasetCard(
            dataset: datasets[index],
            onTap: () => _showDatasetDetail(datasets[index]),
          ),
          childCount: datasets.length,
        ),
      ),
    );
  }

  /// 构建数据集列表（窄屏）- Sliver 版本
  Widget _buildDatasetListSliver(List<DatasetModel> datasets) => SliverPadding(
        padding: ResponsiveUtils.getResponsivePadding(context),
        sliver: SliverList(
          delegate: SliverChildBuilderDelegate(
            (context, index) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: DatasetCard(
                dataset: datasets[index],
                onTap: () => _showDatasetDetail(datasets[index]),
              ),
            ),
            childCount: datasets.length,
          ),
        ),
      );

  /// 构建详情视图
  Widget _buildDetailView(DatasetModel dataset, bool isWide) {
    final imagesAsync = ref.watch(imageProvider);

    return RefreshIndicator(
      onRefresh: () async {
        await ref.read(imageProvider.notifier).refresh();
      },
      child: CustomScrollView(
        slivers: [
          // 页面头部
          _buildHeaderSliver(dataset),

          // 数据集信息卡片
          SliverToBoxAdapter(
            child: Padding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              child: DatasetInfoCard(dataset: dataset),
            ),
          ),

          // 类型筛选标签
          SliverToBoxAdapter(
            child: TypeFilterTabs(
              selectedType: ref.watch(imageTypeFilterProvider),
              onTypeChanged: (type) {
                ref.read(imageTypeFilterProvider.notifier).state = type;
                ref.read(imageProvider.notifier).filterByType(type);
              },
            ),
          ),

          // 图片网格
          imagesAsync.when(
            data: (images) => ImageGrid(
              images: images,
              onImageTap: _showImageViewer,
              asSliver: true,
            ),
            loading: () => const SliverFillRemaining(
              child: Center(child: CircularProgressIndicator()),
            ),
            error: (error, stack) => SliverFillRemaining(
              child: _buildErrorState(error.toString()),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(String message, IconData icon) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 64,
              color: const Color(0xFFD1D5DB), // gray-300
            ),
            const SizedBox(height: 12),
            Text(
              message,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: const Color(0xFF6B7280), // gray-500
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
            const SizedBox(height: 16),
            Text(
              '加载失败',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                error,
                style: Theme.of(context).textTheme.bodyMedium,
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () {
                final selectedDataset = ref.read(selectedDatasetProvider);
                if (selectedDataset != null) {
                  ref.read(imageProvider.notifier).refresh();
                } else {
                  ref.read(datasetProvider.notifier).refresh();
                }
              },
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );

  void _performSearch(String query) {
    final selectedDataset = ref.read(selectedDatasetProvider);

    if (selectedDataset != null) {
      ref.read(imageProvider.notifier).searchImages(query);
    } else {
      ref.read(datasetProvider.notifier).searchDatasets(query);
    }
  }

  void _showDatasetDetail(DatasetModel dataset) {
    ref.read(selectedDatasetProvider.notifier).state = dataset;
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

  /// 显示图片查看器
  void _showImageViewer(ImageModel image) {
    final typeLabels = {
      ImageType.hazy: '有雾图像',
      ImageType.clear: '清晰图像',
      ImageType.dehazed: '去雾结果',
    };

    showDialog<void>(
      context: context,
      barrierColor: Colors.black.withValues(alpha: 0.9),
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.all(16),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 900, maxHeight: 700),
          child: Stack(
            children: [
              // 图片
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: CachedNetworkImage(
                  imageUrl: image.imageUrl,
                  fit: BoxFit.contain,
                  placeholder: (context, url) => Container(
                    color: Colors.grey[900],
                    child: const Center(child: CircularProgressIndicator()),
                  ),
                  errorWidget: (context, url, error) => Container(
                    color: Colors.grey[900],
                    child: const Icon(
                      Icons.broken_image_outlined,
                      color: Colors.white,
                      size: 64,
                    ),
                  ),
                ),
              ),

              // 关闭按钮
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

              // 图片信息
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    borderRadius: const BorderRadius.only(
                      bottomLeft: Radius.circular(12),
                      bottomRight: Radius.circular(12),
                    ),
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
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 16,
                        children: [
                          Text(
                            '类型: ${typeLabels[image.imageType]}',
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.9),
                              fontSize: 14,
                            ),
                          ),
                          Text(
                            '尺寸: ${image.width} × ${image.height}',
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.9),
                              fontSize: 14,
                            ),
                          ),
                          if (image.fileSize != null)
                            Text(
                              '大小: ${_formatFileSize(image.fileSize!)}',
                              style: TextStyle(
                                color: Colors.white.withValues(alpha: 0.9),
                                fontSize: 14,
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _formatFileSize(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
}
