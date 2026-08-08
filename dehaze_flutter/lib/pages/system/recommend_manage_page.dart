import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/recommendation_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 推荐管理页面（L2）
///
/// 权限：sys:recommendation:*
class RecommendManagePage extends ConsumerStatefulWidget {
  const RecommendManagePage({super.key});

  @override
  ConsumerState<RecommendManagePage> createState() => _RecommendManagePageState();
}

class _RecommendManagePageState extends ConsumerState<RecommendManagePage> {
  List<RecommendationRule> _rules = [];
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
  }

  Future<void> _fetchData() async {
    setState(() { _loading = true; _error = null; });
    try {
      final rules = await ref.read(recommendationServiceProvider).getRules();
      if (mounted) setState(() { _rules = rules; _loading = false; });
    } catch (e) {
      if (mounted) setState(() { _error = extractErrorMessage(e); _loading = false; });
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _editRule(int id, RecommendationRule rule) {
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
              if (!mounted) return;
              _showSnack('更新成功');
              _fetchData();
            } catch (e) {
              _showSnack(extractErrorMessage(e));
            }
          }, child: const Text('保存')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:recommendation:*')) {
      return Scaffold(appBar: AppBar(title: const Text('推荐管理')), body: const Center(child: Text('无权限访问')));
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('推荐管理')),
      body: _loading ? const Center(child: CircularProgressIndicator())
          : _error != null ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text(_error!, style: TextStyle(color: theme.colorScheme.error)), SizedBox(height: AppTheme.spacingM),
            FilledButton(onPressed: _fetchData, child: const Text('重试')),
          ]))
          : _rules.isEmpty ? const Center(child: Text('暂无推荐规则'))
          : RefreshIndicator(
            onRefresh: () => _fetchData(),
            child: ListView.builder(
              itemCount: _rules.length,
              itemBuilder: (context, index) {
                final rule = _rules[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.auto_awesome),
                    title: Text(rule.ruleName),
                    subtitle: Text('权重: ${rule.weight} | 状态: ${rule.enabled ? '启用' : '禁用'}'),
                    trailing: IconButton(icon: const Icon(Icons.edit, size: 20), onPressed: () => _editRule(rule.id ?? 0, rule)),
                  ),
                );
              },
            ),
          ),
    );
  }
}
