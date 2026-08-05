import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../providers/processing_provider.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import 'widgets/compare_empty_state.dart';
import 'widgets/comparison_scaffold.dart';

/// 算法信息页面
///
/// 展示算法详情：名称、类型、状态、描述、配置参数、元信息
///
/// 优先展示处理流程中已选中的算法；若无（如从菜单/CTA 直接进入），
/// 自动加载后端算法列表并展示第一个启用的算法，避免空状态。
class AlgorithmInfoPage extends ConsumerStatefulWidget {
  const AlgorithmInfoPage({super.key});

  @override
  ConsumerState<AlgorithmInfoPage> createState() => _AlgorithmInfoPageState();
}

class _AlgorithmInfoPageState extends ConsumerState<AlgorithmInfoPage> {
  AlgorithmModel? _loadedAlgorithm;
  bool _isLoading = false;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _maybeLoadAlgorithm());
  }

  /// 处理流程已选中算法时无需加载；否则拉取算法列表取首个启用项。
  Future<void> _maybeLoadAlgorithm() async {
    if (ref.read(processingProvider).selectedAlgorithm != null) return;
    if (_loadedAlgorithm != null || _isLoading) return;

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final service = ref.read(algorithmServiceProvider);
      final algorithms = await service.getAlgorithmList();

      // 展平树形结构，只取已发布的叶子算法
      final flatAlgorithms = algorithms.flatPublishedLeaves;

      if (!mounted) return;
      setState(() {
        _loadedAlgorithm = flatAlgorithms.isNotEmpty ? flatAlgorithms.first : null;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = extractErrorMessage(e);
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final selected = ref.watch(processingProvider).selectedAlgorithm;
    final algorithm = selected ?? _loadedAlgorithm;

    if (algorithm == null) {
      if (_isLoading) {
        return ComparisonScaffold(
          icon: Icons.info_outline,
          title: '算法信息',
          currentRoute: AppRouterConfig.algorithm,
          body: Center(child: CircularProgressIndicator(color: AppTheme.brandBlue)),
        );
      }
      if (_errorMessage != null) {
        return ComparisonScaffold(
          icon: Icons.info_outline,
          title: '算法信息',
          currentRoute: AppRouterConfig.algorithm,
          body: _buildError(theme),
        );
      }
      return ComparisonScaffold(
        icon: Icons.info_outline,
        title: '算法信息',
        currentRoute: AppRouterConfig.algorithm,
        body: CompareEmptyState(
          icon: Icons.inbox_outlined,
          iconColor: theme.colorScheme.onSurfaceVariant,
          message: '暂无可用算法',
          actionLabel: '重试加载',
          onAction: () => context.go(AppRouterConfig.algorithmSelect),
        ),
      );
    }

    return ComparisonScaffold(
      icon: Icons.info_outline,
      title: '算法信息',
      subtitle: algorithm.name,
      currentRoute: AppRouterConfig.algorithm,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildInfoCard(theme, algorithm),
            const SizedBox(height: 16),
            if (algorithm.description != null) ...[
              _buildDescriptionCard(theme, algorithm.description!),
              const SizedBox(height: 16),
            ],
            _buildMetaCard(theme, algorithm),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard(ThemeData theme, AlgorithmModel algorithm) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('基本信息', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 12),
              _buildInfoRow(theme, '算法名称', algorithm.name),
              _buildInfoRow(theme, '算法类型', algorithm.type),
              _buildInfoRow(theme, '算法状态', algorithm.status.displayName),
            ],
          ),
        ),
      );

  Widget _buildDescriptionCard(ThemeData theme, String description) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('算法描述', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              Text(description, style: theme.textTheme.bodyMedium),
            ],
          ),
        ),
      );

  Widget _buildMetaCard(ThemeData theme, AlgorithmModel algorithm) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('元信息', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 12),
              if (algorithm.path != null)
                _buildInfoRow(theme, '模型路径', algorithm.path!),
            ],
          ),
        ),
      );

  Widget _buildInfoRow(ThemeData theme, String label, String value) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 100,
              child: Text(label, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
            ),
            Expanded(child: Text(value, style: theme.textTheme.bodyMedium)),
          ],
        ),
      );

  Widget _buildError(ThemeData theme) => Center(
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
              onPressed: () {
                setState(() {
                  _loadedAlgorithm = null;
                  _errorMessage = null;
                });
                _maybeLoadAlgorithm();
              },
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );
}
