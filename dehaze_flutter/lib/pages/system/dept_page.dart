import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/dept_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 部门管理页面（L2）
///
/// 权限：sys:dept:*
class DeptManagePage extends ConsumerStatefulWidget {
  const DeptManagePage({super.key});

  @override
  ConsumerState<DeptManagePage> createState() => _DeptManagePageState();
}

class _DeptManagePageState extends ConsumerState<DeptManagePage> {
  List<Dept> _depts = [];
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
      final depts = await ref.read(deptServiceProvider).getList();
      if (mounted) { setState(() { _depts = depts; _loading = false; }); }
    } catch (e) {
      if (mounted) { setState(() { _error = extractErrorMessage(e); _loading = false; }); }
    }
  }

  Future<void> _deleteDept(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(title: const Text('确认删除'), content: const Text('删除部门将同时删除子部门，确定？'), actions: [
        TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('取消')),
        FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('确定')),
      ]),
    );
    if (confirmed != true) { return; }
    try {
      await ref.read(deptServiceProvider).delete(id);
      _showSnack('删除成功');
      _fetchData();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  void _showSnack(String msg) {
    if (!mounted) { return; }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showForm({int? deptId, int? parentId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _DeptFormSheet(deptId: deptId, parentId: parentId, onSaved: () { _fetchData(); Navigator.pop(c); }),
    );
  }

  List<Widget> _buildTree(List<Dept> items, int depth) {
    final widgets = <Widget>[];
    for (final item in items) {
      final children = item.children;
      final status = item.status;
      widgets.add(
        Card(
          child: ListTile(
            contentPadding: EdgeInsets.only(left: 16.0 + depth * 24.0, right: 8),
            leading: Icon(Icons.business, size: 20, color: status == 1 ? AppTheme.successColor : AppTheme.errorColor),
            title: Text(item.name, style: TextStyle(fontWeight: depth == 0 ? FontWeight.w600 : FontWeight.w400)),
            subtitle: Text('排序: ${item.sort}'),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(icon: const Icon(Icons.add, size: 20), tooltip: '添加子部门', onPressed: () => _showForm(parentId: item.id)),
                IconButton(icon: const Icon(Icons.edit, size: 20), tooltip: '编辑', onPressed: () => _showForm(deptId: item.id)),
                IconButton(icon: Icon(Icons.delete, size: 20, color: AppTheme.errorColor), tooltip: '删除', onPressed: () => _deleteDept(item.id)),
              ],
            ),
          ),
        ),
      );
      if (children != null && children.isNotEmpty) {
        widgets.addAll(_buildTree(children, depth + 1));
      }
    }
    return widgets;
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:dept:*')) {
      return Scaffold(appBar: AppBar(title: const Text('部门管理')), body: const Center(child: Text('无权限访问')));
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('部门管理'),
        actions: [
          if (auth.hasPerm('sys:dept:add')) IconButton(icon: const Icon(Icons.add), tooltip: '新增部门', onPressed: () => _showForm()),
        ],
      ),
      body: _loading ? const Center(child: CircularProgressIndicator())
          : _error != null ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
            Text(_error!, style: TextStyle(color: theme.colorScheme.error)), SizedBox(height: AppTheme.spacingM),
            FilledButton(onPressed: _fetchData, child: const Text('重试')),
          ]))
          : _depts.isEmpty ? const Center(child: Text('暂无数据'))
          : RefreshIndicator(onRefresh: () => _fetchData(), child: ListView(children: _buildTree(_depts, 0))),
    );
  }
}

class _DeptFormSheet extends ConsumerStatefulWidget {
  const _DeptFormSheet({this.deptId, this.parentId, required this.onSaved});
  final int? deptId;
  final int? parentId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_DeptFormSheet> createState() => _DeptFormSheetState();
}

class _DeptFormSheetState extends ConsumerState<_DeptFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _sortCtrl = TextEditingController();
  int _status = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.deptId != null) { _loadData(); }
  }

  @override
  void dispose() {
    _nameCtrl.dispose(); _sortCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref.read(deptServiceProvider).getById(widget.deptId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name;
          _sortCtrl.text = data.sort.toString();
          _status = data.status;
        });
      }
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) { return; }
    setState(() => _loading = true);
    try {
      final form = DeptForm(
        parentId: widget.parentId ?? 0,
        name: _nameCtrl.text.trim(),
        sort: int.tryParse(_sortCtrl.text) ?? 0,
        status: _status,
      );
      if (widget.deptId != null) {
        await ref.read(deptServiceProvider).update(widget.deptId!, form);
      } else {
        await ref.read(deptServiceProvider).add(form);
      }
      widget.onSaved();
    } catch (e) {
      if (mounted) { ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(extractErrorMessage(e)))); }
    } finally {
      if (mounted) { setState(() => _loading = false); }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isNew = widget.deptId == null;
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: SingleChildScrollView(
        padding: EdgeInsets.all(AppTheme.spacingL),
        child: Form(key: _formKey, child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text(isNew ? '新增部门' : '编辑部门', style: Theme.of(context).textTheme.titleLarge),
          SizedBox(height: AppTheme.spacingL),
          TextFormField(controller: _nameCtrl, decoration: const InputDecoration(labelText: '部门名称'), validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null),
          SizedBox(height: AppTheme.spacingM),
          TextFormField(controller: _sortCtrl, decoration: const InputDecoration(labelText: '排序'), keyboardType: TextInputType.number),
          SizedBox(height: AppTheme.spacingL),
          SizedBox(height: 44, child: FilledButton(onPressed: _loading ? null : _submit, child: Text(_loading ? '提交中...' : '保存'))),
        ])),
      ),
    );
  }
}
