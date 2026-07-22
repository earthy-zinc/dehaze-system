import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
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
      final flatAlgorithms = <AlgorithmModel>[];
      for (final algo in algorithms) {
        if (algo.children.isEmpty && algo.isEnabled) {
          flatAlgorithms.add(algo);
        } else {
          for (final child in algo.children) {
            if (child.isEnabled) {
              flatAlgorithms.add(child);
            }
          }
        }
      }

      if (mounted) {
        setState(() {
          _algorithms = flatAlgorithms;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = _extractErrorMessage(e);
          _isLoading = false;
        });
      }
    }
  }

  String _extractErrorMessage(dynamic e) {
    if (e is ApiException) return e.message;
    return e.toString().replaceFirst('Exception: ', '');
  }

  /// 下一步：上传选中的图片并进入去雾处理
  ///
  /// 打通 图像输入 → 算法选择 → 去雾处理 的关键衔接：
  /// 将图像输入页选中的图片字节流上传到文件服务，
  /// 拿到 fileId 后设置到处理流程，再跳转处理页。
  Future<void> _uploadAndProceed() async {
    final selectedImage = ref.read(selectedImageProvider);
    if (selectedImage == null) {
      _showSnackBar('请先选择图片');
      return;
    }

    final bytes = selectedImage.bytes;
    if (bytes == null || bytes.isEmpty) {
      _showSnackBar('图片数据无效，请重新选择');
      return;
    }

    setState(() => _isUploading = true);

    try {
      final fileService = ref.read(fileServiceProvider);
      final uploadResult = await fileService.uploadBytes(
        bytes,
        selectedImage.filename,
      );

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
        _showSnackBar('图片上传失败: ${_extractErrorMessage(e)}');
      }
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        behavior: SnackBarBehavior.floating,
      ),
    );
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
