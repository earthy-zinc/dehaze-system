import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/algorithm_model.dart';
import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 算法信息页面
///
/// 展示算法详情：名称、类型、状态、描述、配置参数、元信息
class AlgorithmInfoPage extends ConsumerWidget {
  const AlgorithmInfoPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);
    final algorithm = state.selectedAlgorithm;

    if (algorithm == null) {
      return _buildNoData(context, theme);
    }

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme, algorithm.name),
          Expanded(
            child: SingleChildScrollView(
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
                  if (algorithm.config != null && algorithm.config!.isNotEmpty) ...[
                    _buildConfigCard(theme, algorithm.config!),
                    const SizedBox(height: 16),
                  ],
                  _buildMetaCard(theme, algorithm),
                ],
              ),
            ),
          ),
          _buildBottomNav(context),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme, String name) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(bottom: BorderSide(color: theme.dividerColor)),
        ),
        child: Row(
          children: [
            Icon(Icons.info_outline, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('算法信息', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(width: 16),
            Text(name, style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );

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
              if (algorithm.remark != null)
                _buildInfoRow(theme, '备注', algorithm.remark!),
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

  Widget _buildConfigCard(ThemeData theme, Map<String, dynamic> config) => Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('配置参数', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              ...config.entries.map((entry) => _buildInfoRow(theme, entry.key, entry.value.toString())),
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
              if (algorithm.createTime != null)
                _buildInfoRow(theme, '创建时间', algorithm.createTime!),
              if (algorithm.updateTime != null)
                _buildInfoRow(theme, '更新时间', algorithm.updateTime!),
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

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('并排对比'), onPressed: () => context.go(AppRouterConfig.sideBySide)),
            ActionChip(label: const Text('重叠对比'), onPressed: () => context.go(AppRouterConfig.overlay)),
            ActionChip(label: const Text('放大镜'), onPressed: () => context.go(AppRouterConfig.magnifier)),
            ActionChip(label: const Text('滤镜调节'), onPressed: () => context.go(AppRouterConfig.filter)),
            ActionChip(label: const Text('指标评估'), onPressed: () => context.go(AppRouterConfig.metrics)),
          ],
        ),
      );

  Widget _buildNoData(BuildContext context, ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.warning_amber, size: 64, color: theme.colorScheme.error),
            const SizedBox(height: 16),
            Text('请先选择算法', style: theme.textTheme.titleMedium),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => context.go(AppRouterConfig.algorithmSelect),
              child: const Text('去选择'),
            ),
          ],
        ),
      );
}
