import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../models/file_model.dart';
import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import '../../utils/ui_utils.dart';
import '../image_input/models/image_input_model.dart';
import '../image_input/providers/image_input_provider.dart';

/// 算法选择页面
class AlgorithmSelectPage extends ConsumerStatefulWidget {
  const AlgorithmSelectPage({super.key});

  @override
  ConsumerState<AlgorithmSelectPage> createState() =>
      _AlgorithmSelectPageState();
}

class _AlgorithmSelectPageState extends ConsumerState<AlgorithmSelectPage> {
  List<AlgorithmModel> _algorithms = [];
  bool _isLoading = true;
  bool _isUploading = false;
  String? _errorMessage;
  String _searchQuery = '';

  List<AlgorithmRecommend> _recommendations = [];
  bool _isRecommending = false;
  String? _recommendError;
  FileUploadResponse? _uploadedFile;

  List<AlgorithmModel> get _filteredAlgorithms {
    final query = _searchQuery.trim().toLowerCase();
    if (query.isEmpty) return _algorithms;
    return _algorithms.where((algo) {
      final name = algo.name.toLowerCase();
      final desc = (algo.description ?? '').toLowerCase();
      return name.contains(query) || desc.contains(query);
    }).toList();
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadAlgorithms());
  }

  Future<void> _loadAlgorithms() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final service = ref.read(algorithmServiceProvider);
      final algorithms = await service.getAlgorithmList();

      // 展平树形结构，只显示启用的叶子算法
      final flatAlgorithms = algorithms.flatEnabledLeaves;

      if (mounted) {
        setState(() {
          _algorithms = flatAlgorithms;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = extractErrorMessage(e);
          _isLoading = false;
        });
      }
    }
  }

  /// 下一步：上传选中的图片并进入去雾处理
  ///
  /// 打通 图像输入 → 算法选择 → 去雾处理 的关键衔接：
  /// 将图像输入页选中的图片字节流上传到文件服务，
  /// 拿到 fileId 后设置到处理流程，再跳转处理页。
  Future<void> _uploadAndProceed() async {
    final selectedImage = ref.read(selectedImageProvider);
    if (selectedImage == null) {
      showSnackBar(context, '请先选择图片');
      return;
    }

    final bytes = selectedImage.bytes;
    if (bytes == null || bytes.isEmpty) {
      showSnackBar(context, '图片数据无效，请重新选择');
      return;
    }

    setState(() => _isUploading = true);

    try {
      final uploadResult = await _ensureUploaded(selectedImage);

      if (!mounted) return;

      // 构造处理流程所需的 SelectedImage 并设置
      ref.read(processingProvider.notifier).setSelectedImage(SelectedImage(
            fileId: uploadResult.id,
            fileUrl: uploadResult.url,
            fileName: uploadResult.name,
            bytes: bytes,
            cleanUrl: selectedImage.sampleInfo?.cleanUrl,
          ));

      context.go(AppRouterConfig.processing);
    } catch (e) {
      if (mounted) {
        showSnackBar(context, '图片上传失败: ${extractErrorMessage(e)}');
      }
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  /// 确保图片已上传，返回上传结果（带缓存，避免重复上传）
  Future<FileUploadResponse> _ensureUploaded(
      SelectedImageModel selectedImage) async {
    if (_uploadedFile != null) return _uploadedFile!;
    final fileService = ref.read(fileServiceProvider);
    _uploadedFile = await fileService.uploadBytes(
      selectedImage.bytes!,
      selectedImage.filename,
    );
    return _uploadedFile!;
  }

  /// 加载智能推荐算法
  Future<void> _loadRecommendations() async {
    final selectedImage = ref.read(selectedImageProvider);
    if (selectedImage == null) {
      showSnackBar(context, '请先选择图片');
      return;
    }

    setState(() {
      _isRecommending = true;
      _recommendError = null;
    });

    try {
      // 样例图片有可访问的远程 URL，直接使用；
      // 上传/拍照图片需先上传获取可访问 URL
      String imageUrl;
      if (selectedImage.url.startsWith('http')) {
        imageUrl = selectedImage.url;
      } else {
        final uploadResult = await _ensureUploaded(selectedImage);
        imageUrl = uploadResult.url;
      }

      final service = ref.read(algorithmServiceProvider);
      final recommendations =
          await service.recommendAlgorithms(imageUrl: imageUrl, topN: 3);

      if (mounted) {
        setState(() {
          _recommendations = recommendations;
          _isRecommending = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _recommendError = extractErrorMessage(e);
          _isRecommending = false;
        });
      }
    }
  }

  /// 通过推荐结果选中对应算法
  void _selectRecommendation(AlgorithmRecommend recommend) {
    final algorithm =
        _algorithms.where((a) => a.id == recommend.algorithmId).firstOrNull;
    if (algorithm != null) {
      ref.read(processingProvider.notifier).setSelectedAlgorithm(algorithm);
      showSnackBar(context, '已选择: ${algorithm.name}');
    } else {
      showSnackBar(context, '未找到算法: ${recommend.algorithmName}');
    }
  }

  @override
  Widget build(BuildContext context) {
    final processingState = ref.watch(processingProvider);
    final selectedAlgorithm = processingState.selectedAlgorithm;
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1000,
        child: CustomScrollView(
          slivers: [
            // 页面头部
            SliverToBoxAdapter(child: _buildHeader(theme)),

            // 搜索框
            SliverToBoxAdapter(child: _buildSearchBar(theme)),

            // 智能推荐区域
            SliverToBoxAdapter(child: _buildRecommendationSection(theme)),

            // 内容区域
            SliverPadding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              sliver: _buildContent(theme, selectedAlgorithm),
            ),
          ],
        ),
      ),
      bottomNavigationBar: selectedAlgorithm != null
          ? _buildBottomBar(theme, selectedAlgorithm)
          : null,
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.psychology_outlined,
                    color: AppTheme.brandBlue, size: 24),
                const SizedBox(width: 8),
                Text(
                  '选择算法',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '选择适合的去雾算法进行处理',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );

  Widget _buildSearchBar(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        color: theme.colorScheme.surface,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 12),
            TextField(
              onChanged: (value) => setState(() => _searchQuery = value),
              decoration: InputDecoration(
                hintText: '搜索算法名称或描述',
                prefixIcon: const Icon(Icons.search_outlined),
                suffixIcon: _searchQuery.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () =>
                            setState(() => _searchQuery = ''),
                      )
                    : null,
              ),
            ),
            const SizedBox(height: 12),
          ],
        ),
      );

  Widget _buildRecommendationSection(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(Icons.auto_awesome,
                    color: AppTheme.techGreen, size: 20),
                const SizedBox(width: 8),
                Text(
                  '智能推荐',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                TextButton.icon(
                  onPressed: _isRecommending ? null : _loadRecommendations,
                  icon: _isRecommending
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.refresh, size: 18),
                  label: const Text('获取推荐'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (_recommendError != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(
                  _recommendError!,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: theme.colorScheme.error),
                ),
              ),
            if (_recommendations.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Column(
                  children: _recommendations
                      .map((r) => _RecommendationCard(
                            recommend: r,
                            onTap: () => _selectRecommendation(r),
                          ))
                      .toList(),
                ),
              )
            else if (!_isRecommending && _recommendError == null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Text(
                  '选择图片后点击"获取推荐"，系统将根据图片特征推荐最合适的算法',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
          ],
        ),
      );

  Widget _buildContent(ThemeData theme, AlgorithmModel? selected) {
    if (_isLoading) {
      return const SliverFillRemaining(
        child: Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return SliverFillRemaining(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, size: 64, color: theme.colorScheme.error),
              const SizedBox(height: 16),
              Text('加载失败', style: theme.textTheme.titleLarge),
              const SizedBox(height: 8),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Text(
                  _errorMessage!,
                  style: theme.textTheme.bodyMedium,
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _loadAlgorithms,
                icon: const Icon(Icons.refresh),
                label: const Text('重试'),
              ),
            ],
          ),
        ),
      );
    }

    if (_algorithms.isEmpty) {
      return const SliverFillRemaining(
        child: Center(child: Text('暂无可用算法')),
      );
    }

    final filtered = _filteredAlgorithms;

    if (filtered.isEmpty) {
      return const SliverFillRemaining(
        child: Center(child: Text('未找到匹配的算法')),
      );
    }

    return SliverList(
      delegate: SliverChildBuilderDelegate(
        (context, index) {
          final algorithm = filtered[index];
          final isSelected = selected?.id == algorithm.id;
          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: _AlgorithmCard(
              algorithm: algorithm,
              isSelected: isSelected,
              onTap: () {
                ref
                    .read(processingProvider.notifier)
                    .setSelectedAlgorithm(algorithm);
              },
            ),
          );
        },
        childCount: filtered.length,
      ),
    );
  }

  Widget _buildBottomBar(ThemeData theme, AlgorithmModel selected) =>
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            top: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: SafeArea(
          child: Row(
            children: [
              Expanded(
                child: Text(
                  '已选择: ${selected.name}',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              FilledButton.icon(
                onPressed: _isUploading ? null : _uploadAndProceed,
                icon: _isUploading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.arrow_forward),
                label: Text(_isUploading ? '上传中...' : '下一步'),
              ),
            ],
          ),
        ),
      );
}

/// 算法卡片
class _AlgorithmCard extends StatelessWidget {
  const _AlgorithmCard({
    required this.algorithm,
    required this.isSelected,
    required this.onTap,
  });

  final AlgorithmModel algorithm;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Material(
      color: isSelected
          ? AppTheme.brandBlue.withValues(alpha: 0.05)
          : theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            border: Border.all(
              color: isSelected
                  ? AppTheme.brandBlue
                  : theme.colorScheme.outline,
              width: isSelected ? 2 : 1,
            ),
          ),
          child: Row(
            children: [
              // 图标
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: algorithm.isDeepLearning
                      ? AppTheme.techGreen.withValues(alpha: 0.1)
                      : AppTheme.brandBlue.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  algorithm.isDeepLearning
                      ? Icons.memory
                      : Icons.auto_fix_high,
                  color: algorithm.isDeepLearning
                      ? AppTheme.techGreen
                      : AppTheme.brandBlue,
                ),
              ),
              const SizedBox(width: 16),
              // 信息
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          algorithm.name,
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: algorithm.isDeepLearning
                                ? AppTheme.techGreen.withValues(alpha: 0.1)
                                : AppTheme.brandBlue.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            algorithm.type,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w500,
                              color: algorithm.isDeepLearning
                                  ? AppTheme.techGreen
                                  : AppTheme.brandBlue,
                            ),
                          ),
                        ),
                      ],
                    ),
                    if (algorithm.description != null) ...[
                      const SizedBox(height: 4),
                      Text(
                        algorithm.description!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ],
                ),
              ),
              // 选中标记
              if (isSelected)
                Icon(Icons.check_circle, color: AppTheme.brandBlue, size: 24),
            ],
          ),
        ),
      ),
    );
  }
}

/// 推荐算法卡片
class _RecommendationCard extends StatelessWidget {
  const _RecommendationCard({
    required this.recommend,
    required this.onTap,
  });

  final AlgorithmRecommend recommend;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Material(
        color: AppTheme.techGreen.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(AppTheme.radiusM),
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(AppTheme.radiusM),
              border: Border.all(
                color: AppTheme.techGreen.withValues(alpha: 0.3),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AppTheme.techGreen.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.auto_awesome,
                      color: AppTheme.techGreen, size: 20),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              recommend.algorithmName,
                              style: theme.textTheme.titleSmall?.copyWith(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 2,
                            ),
                            decoration: BoxDecoration(
                              color: AppTheme.techGreen.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              '匹配度 ${recommend.score.toStringAsFixed(0)}%',
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w500,
                                color: AppTheme.techGreen,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        recommend.reason,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Icon(Icons.chevron_right,
                    color: theme.colorScheme.onSurfaceVariant),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
