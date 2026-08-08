import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

/// 算法库浏览页面（L2，ToolsStack 内）
///
/// 浏览所有已发布算法，支持搜索、查看详情、「使用该算法」跳转算法选择页。
class AlgorithmBrowsePage extends ConsumerStatefulWidget {
  const AlgorithmBrowsePage({super.key});

  @override
  ConsumerState<AlgorithmBrowsePage> createState() =>
      _AlgorithmBrowsePageState();
}

class _AlgorithmBrowsePageState extends ConsumerState<AlgorithmBrowsePage> {
  List<AlgorithmModel> _algorithms = [];
  bool _isLoading = true;
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
      final algorithms = await service.getList();
      final flatAlgorithms = algorithms.flatPublishedLeaves;

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

  /// 跳转到去雾流程的算法选择页，传递选中的算法 ID
  void _useAlgorithm(AlgorithmModel algorithm) {
    context.go(AppRouterConfig.algorithmSelect, extra: algorithm.id);
  }

  /// 显示算法详情弹窗
  void _showAlgorithmDetail(AlgorithmModel algorithm) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => _AlgorithmDetailSheet(algorithm: algorithm),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1000,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildHeader(theme)),
            SliverToBoxAdapter(child: _buildSearchBar(theme)),
            SliverPadding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              sliver: _buildContent(theme),
            ),
          ],
        ),
      ),
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
                  '算法库',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '浏览全部去雾算法，选择合适的算法开始处理',
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
                        onPressed: () => setState(() => _searchQuery = ''),
                      )
                    : null,
              ),
            ),
            const SizedBox(height: 12),
          ],
        ),
      );

  Widget _buildContent(ThemeData theme) {
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
          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: _BrowseAlgorithmCard(
              algorithm: algorithm,
              onTap: () => _showAlgorithmDetail(algorithm),
              onUse: () => _useAlgorithm(algorithm),
            ),
          );
        },
        childCount: filtered.length,
      ),
    );
  }
}

/// 浏览版算法卡片
class _BrowseAlgorithmCard extends StatelessWidget {
  const _BrowseAlgorithmCard({
    required this.algorithm,
    required this.onTap,
    required this.onUse,
  });

  final AlgorithmModel algorithm;
  final VoidCallback onTap;
  final VoidCallback onUse;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Material(
      color: theme.colorScheme.surface,
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            border: Border.all(color: theme.colorScheme.outline),
          ),
          child: Row(
            children: [
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
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Flexible(
                          child: Text(
                            algorithm.name,
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
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
              const SizedBox(width: 8),
              FilledButton.tonalIcon(
                onPressed: onUse,
                icon: const Icon(Icons.play_arrow, size: 18),
                label: const Text('使用'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// 算法详情底部弹窗
class _AlgorithmDetailSheet extends StatelessWidget {
  const _AlgorithmDetailSheet({required this.algorithm});

  final AlgorithmModel algorithm;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // 捕获祖先 Router 上下文以在弹窗中使用
    final router = GoRouter.of(context);

    return DraggableScrollableSheet(
      initialChildSize: 0.6,
      minChildSize: 0.3,
      maxChildSize: 0.85,
      expand: false,
      builder: (ctx, scrollController) => Container(
        padding: const EdgeInsets.all(24),
        child: ListView(
          controller: scrollController,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: algorithm.isDeepLearning
                        ? AppTheme.techGreen.withValues(alpha: 0.1)
                        : AppTheme.brandBlue.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(
                    algorithm.isDeepLearning
                        ? Icons.memory
                        : Icons.auto_fix_high,
                    color: algorithm.isDeepLearning
                        ? AppTheme.techGreen
                        : AppTheme.brandBlue,
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        algorithm.name,
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
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
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                color: algorithm.isDeepLearning
                                    ? AppTheme.techGreen
                                    : AppTheme.brandBlue,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          if (algorithm.isDeepLearning)
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 2,
                              ),
                              decoration: BoxDecoration(
                                color: AppTheme.techGreen.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: const Text(
                                '深度学习',
                                style: TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w500,
                                  color: AppTheme.techGreen,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            if (algorithm.description != null) ...[
              _sectionTitle(theme, '算法描述'),
              const SizedBox(height: 8),
              Text(
                algorithm.description!,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurface,
                  height: 1.6,
                ),
              ),
              const SizedBox(height: 24),
            ],
            if (algorithm.path != null) ...[
              _sectionTitle(theme, '模型文件'),
              const SizedBox(height: 8),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest,
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                ),
                child: Row(
                  children: [
                    Icon(Icons.folder_outlined,
                        size: 18,
                        color: theme.colorScheme.onSurfaceVariant),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        algorithm.path!,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                          fontFamily: 'monospace',
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
            ],
            const SizedBox(height: 8),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: () {
                  Navigator.of(context).pop();
                  router.go(AppRouterConfig.algorithmSelect,
                      extra: algorithm.id);
                },
                icon: const Icon(Icons.play_arrow),
                label: const Text('使用该算法'),
                style: FilledButton.styleFrom(
                  minimumSize: const Size(double.infinity, 48),
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _sectionTitle(ThemeData theme, String title) => Text(
        title,
        style: theme.textTheme.titleMedium?.copyWith(
          fontWeight: FontWeight.w600,
        ),
      );
}
