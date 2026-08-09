import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/recommendation_model.dart';
import '../../providers/providers.dart';
import '../../services/recommendation_service.dart';
import '../../theme/app_theme.dart';

final recommendManageProvider = StateNotifierProvider<
    RecommendManageNotifier, AsyncValue<List<RecommendationRule>>>(
  (ref) => RecommendManageNotifier(ref.watch(recommendationServiceProvider)),
);

class RecommendManageNotifier
    extends StateNotifier<AsyncValue<List<RecommendationRule>>> {
  RecommendManageNotifier(this._service) : super(const AsyncValue.loading()) {
    loadData();
  }

  final RecommendationService _service;

  Future<void> loadData() async {
    state = const AsyncValue.loading();
    try {
      final rules = await _service.getRules();
      if (!mounted) return;
      state = AsyncValue.data(rules);
    } catch (e, st) {
      if (!mounted) return;
      state = AsyncValue.error(e, st);
    }
  }
}

/// 推荐管理页面（L2）
///
/// 权限：sys:recommendation:*
class RecommendManagePage extends ConsumerWidget {
  const RecommendManagePage({super.key});

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _editRule(BuildContext context, WidgetRef ref, int id, RecommendationRule rule) {
    final weightCtrl = TextEditingController(text: rule.weight.toString());
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: Text('编辑规则: ${rule.ruleName}'),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(controller: weightCtrl, decoration: const InputDecoration(labelText: '权重'), keyboardType: TextInputType.number),
        ]),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c), child: const Text('取消')),
          FilledButton(onPressed: () async {
            final updatedRule = RecommendationRule(
              id: rule.id,
              ruleName: rule.ruleName,
              sceneType: rule.sceneType,
              algorithmIds: rule.algorithmIds,
              weight: int.tryParse(weightCtrl.text) ?? 0,
              enabled: rule.enabled,
            );
            Navigator.pop(c);
            try {
              await ref.read(recommendationServiceProvider).updateRule(id, updatedRule);
              if (!context.mounted) return;
              _showSnack(context, '更新成功');
              ref.read(recommendManageProvider.notifier).loadData();
            } catch (e) {
              _showSnack(context, extractErrorMessage(e));
            }
          }, child: const Text('保存')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final state = ref.watch(recommendManageProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('推荐管理')),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
          Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
          SizedBox(height: AppTheme.spacingM),
          FilledButton(
            onPressed: () => ref.read(recommendManageProvider.notifier).loadData(),
            child: const Text('重试')),
        ])),
        data: (rules) => rules.isEmpty
            ? const Center(child: Text('暂无推荐规则'))
            : RefreshIndicator(
                onRefresh: () => ref.read(recommendManageProvider.notifier).loadData(),
                child: ListView.builder(
                  itemCount: rules.length,
                  itemBuilder: (context, index) {
                    final rule = rules[index];
                    return Card(
                      child: ListTile(
                        leading: const Icon(Icons.auto_awesome),
                        title: Text(rule.ruleName),
                        subtitle: Text('权重: ${rule.weight} | 状态: ${rule.enabled ? '启用' : '禁用'}'),
                        trailing: IconButton(
                          icon: const Icon(Icons.edit, size: 20),
                          onPressed: () => _editRule(context, ref, rule.id ?? 0, rule)),
                      ),
                    );
                  },
                ),
              ),
      ),
    );
  }
}
