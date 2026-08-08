import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 算法管理页面（L2）
///
/// 权限：sys:algorithm:*
class AlgorithmManagePage extends ConsumerStatefulWidget {
  const AlgorithmManagePage({super.key});

  @override
  ConsumerState<AlgorithmManagePage> createState() =>
      _AlgorithmManagePageState();
}

class _AlgorithmManagePageState extends ConsumerState<AlgorithmManagePage> {
  final _searchController = TextEditingController();
  List<AlgorithmModel> _items = [];
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _fetchData() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final items = await ref
          .read(algorithmServiceProvider)
          .getList(keywords: _searchController.text);
      if (mounted) {
        setState(() {
          _items = items;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = extractErrorMessage(e);
          _loading = false;
        });
      }
    }
  }

  Future<void> _toggleStatus(int id, AlgorithmStatus currentStatus) async {
    final newStatus =
        currentStatus == AlgorithmStatus.disabled
            ? AlgorithmStatus.published
            : AlgorithmStatus.disabled;
    try {
      await ref
          .read(algorithmServiceProvider)
          .updateStatus(id, newStatus.index);
      _showSnack('状态已更新');
      _fetchData();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _audit(int id) async {
    final approved = await showDialog<bool>(
      context: context,
      builder:
          (c) => AlertDialog(
            title: const Text('审核算法'),
            content: const Text('审核通过？'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(c, false),
                child: const Text('驳回'),
              ),
              FilledButton(
                onPressed: () => Navigator.pop(c, true),
                child: const Text('通过'),
              ),
            ],
          ),
    );
    if (approved == null) {
      return;
    }
    try {
      await ref.read(algorithmServiceProvider).auditAlgorithm(
        id,
        AlgorithmAuditForm(
          approved: approved,
          remark: approved ? '审核通过' : '审核驳回',
        ),
      );
      _showSnack('审核完成');
      _fetchData();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showForm({int? algoId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder:
          (c) => _AlgorithmFormSheet(
            algoId: algoId,
            onSaved: () {
              _fetchData();
              Navigator.pop(c);
            },
          ),
    );
  }

  static const _statusLabels = ['草稿', '待审核', '已审核', '已发布', '已下架', '已废弃'];

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:algorithm:*')) {
      return Scaffold(
        appBar: AppBar(title: const Text('算法管理')),
        body: const Center(child: Text('无权限访问')),
      );
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('算法管理'),
        actions: [
          if (auth.hasPerm('sys:algorithm:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增算法',
              onPressed: () => _showForm(),
            ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(AppTheme.spacingM),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: '搜索算法名称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    _fetchData();
                  },
                ),
              ),
              onSubmitted: (_) => _fetchData(),
            ),
          ),
          Expanded(child: _buildList(theme, auth)),
        ],
      ),
    );
  }

  Widget _buildList(ThemeData theme, AuthState auth) {
    if (_loading && _items.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(onPressed: _fetchData, child: const Text('重试')),
          ],
        ),
      );
    }
    if (_items.isEmpty) {
      return const Center(child: Text('暂无数据'));
    }

    return RefreshIndicator(
      onRefresh: () => _fetchData(),
      child: ListView.builder(
        itemCount: _items.length,
        itemBuilder: (context, index) {
          final item = _items[index];
          final statusIdx = item.status.index;
          final label =
              statusIdx >= 0 && statusIdx < _statusLabels.length
                  ? _statusLabels[statusIdx]
                  : '未知';
          return Card(
            child: ListTile(
              leading: CircleAvatar(
                child: Text(item.name[0].toUpperCase()),
              ),
              title: Text(item.name),
              subtitle: Text('版本: ${item.version ?? '-'} | 状态: $label'),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (auth.hasPerm('sys:algorithm:audit') &&
                      item.status == AlgorithmStatus.pendingAudit)
                    IconButton(
                      icon: const Icon(
                        Icons.fact_check,
                        size: 20,
                        color: AppTheme.infoColor,
                      ),
                      tooltip: '审核',
                      onPressed: () => _audit(item.id),
                    ),
                  IconButton(
                    icon: Icon(
                      item.status == AlgorithmStatus.disabled
                          ? Icons.visibility_off
                          : Icons.visibility,
                      size: 20,
                    ),
                    tooltip:
                        item.status == AlgorithmStatus.disabled ? '下架' : '上架',
                    onPressed: () => _toggleStatus(item.id, item.status),
                  ),
                  IconButton(
                    icon: const Icon(Icons.edit, size: 20),
                    tooltip: '编辑',
                    onPressed: () => _showForm(algoId: item.id),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

class _AlgorithmFormSheet extends ConsumerStatefulWidget {
  const _AlgorithmFormSheet({this.algoId, required this.onSaved});
  final int? algoId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_AlgorithmFormSheet> createState() =>
      _AlgorithmFormSheetState();
}

class _AlgorithmFormSheetState extends ConsumerState<_AlgorithmFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _typeCtrl = TextEditingController();
  final _versionCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _pathCtrl = TextEditingController();
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.algoId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _typeCtrl.dispose();
    _versionCtrl.dispose();
    _descCtrl.dispose();
    _pathCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref
          .read(algorithmServiceProvider)
          .getAlgorithmInfoById(widget.algoId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name;
          _typeCtrl.text = data.type;
          _versionCtrl.text = data.version ?? '';
          _descCtrl.text = data.description ?? '';
          _pathCtrl.text = data.path ?? '';
        });
      }
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    final data = {
      'name': _nameCtrl.text.trim(),
      'type': _typeCtrl.text.trim(),
      'version': _versionCtrl.text.trim(),
      'description': _descCtrl.text.trim(),
      if (_pathCtrl.text.isNotEmpty) 'path': _pathCtrl.text.trim(),
    };
    try {
      final service = ref.read(algorithmServiceProvider);
      if (widget.algoId != null) {
        await service.update(widget.algoId!, data);
      } else {
        await service.add(data);
      }
      widget.onSaved();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(extractErrorMessage(e))),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isNew = widget.algoId == null;
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: SingleChildScrollView(
        padding: EdgeInsets.all(AppTheme.spacingL),
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                isNew ? '新增算法' : '编辑算法',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '算法名称'),
                validator:
                    (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _typeCtrl,
                decoration: const InputDecoration(labelText: '算法类型'),
                validator:
                    (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _versionCtrl,
                decoration: const InputDecoration(labelText: '版本号 (如 v1.0.0)'),
                validator:
                    (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _descCtrl,
                decoration: const InputDecoration(labelText: '描述'),
                maxLines: 3,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _pathCtrl,
                decoration: const InputDecoration(labelText: '路径'),
              ),
              SizedBox(height: AppTheme.spacingL),
              SizedBox(
                height: 44,
                child: FilledButton(
                  onPressed: _loading ? null : _submit,
                  child: Text(_loading ? '提交中...' : '保存'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
