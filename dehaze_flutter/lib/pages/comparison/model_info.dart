import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/ai_model_model.dart';
import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import 'widgets/compare_empty_state.dart';
import 'widgets/comparison_scaffold.dart';

/// 模型信息页面
///
/// 展示处理模型的标识、名称与类型。
///
/// 优先展示处理流程中已选中的模型；若无（如从菜单/CTA 直接进入），
/// 取启用模型列表的首项，避免空状态。
class ModelInfoPage extends ConsumerWidget {
  const ModelInfoPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final selected = ref.watch(processingProvider).selectedModel;
    final modelsAsync = ref.watch(imageEditModelsProvider);
    final model = selected ?? modelsAsync.valueOrNull?.firstOrNull;

    return ComparisonScaffold(
      icon: Icons.info_outline,
      title: '模型信息',
      subtitle: model?.displayName,
      currentRoute: AppRouterConfig.modelInfo,
      body: switch (model) {
        null => modelsAsync.when(
          loading: () => Center(
            child: CircularProgressIndicator(color: AppTheme.brandBlue),
          ),
          error: (error, _) => Center(
            child: Padding(
              padding: const EdgeInsets.all(32),
              child: Text(
                '模型列表加载失败：$error',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ),
          ),
          data: (_) => CompareEmptyState(
            icon: Icons.inbox_outlined,
            iconColor: theme.colorScheme.onSurfaceVariant,
            message: '暂无可用模型',
            actionLabel: '去处理',
            onAction: () => context.go(AppRouterConfig.processing),
          ),
        ),
        _ => _buildBody(theme, model),
      },
    );
  }

  Widget _buildBody(ThemeData theme, AiModelVO model) => SingleChildScrollView(
    padding: const EdgeInsets.all(16),
    child: Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '基本信息',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 12),
            _buildInfoRow(theme, '模型名称', model.displayName),
            _buildInfoRow(theme, '模型标识', model.modelId),
            if (model.modelType != null)
              _buildInfoRow(theme, '模型类型', model.modelType!),
          ],
        ),
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
          child: Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
        Expanded(child: Text(value, style: theme.textTheme.bodyMedium)),
      ],
    ),
  );
}
