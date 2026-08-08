import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/menu_model.dart';
import '../../models/role_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
// services are accessed via providers from providers.dart
import '../../theme/app_theme.dart';

/// 角色管理页面（L2）
///
/// 权限：sys:role:*
class RoleManagePage extends ConsumerStatefulWidget {
  const RoleManagePage({super.key});

  @override
  ConsumerState<RoleManagePage> createState() => _RoleManagePageState();
}

class _RoleManagePageState extends ConsumerState<RoleManagePage> {
  final _searchController = TextEditingController();
  List<RolePageVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
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

  Future<void> _fetchData({bool reset = false}) async {
    if (reset) {
      _pageNum = 1;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final result = await ref.read(roleServiceProvider).getPage(
            RoleQuery(
              pageNum: _pageNum,
              pageSize: 10,
              keywords: _searchController.text,
            ),
          );
      setState(() {
        if (reset) {
          _items = result.list;
        } else {
          _items.addAll(result.list);
        }
        _total = result.total;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = extractErrorMessage(e);
        _loading = false;
      });
    }
  }

  Future<void> _deleteRole(int roleId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('确认删除'),
        content: const Text('确定要删除该角色吗？'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(c, false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(c, true),
            child: const Text('确定'),
          ),
        ],
      ),
    );
    if (confirmed != true) {
      return;
    }
    try {
      await ref.read(roleServiceProvider).deleteByIds([roleId]);
      _showSnack('删除成功');
      _fetchData(reset: true);
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showPermissionDialog(int roleId, String roleName) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _RolePermissionSheet(
        roleId: roleId,
        roleName: roleName,
        onSaved: () {
          _fetchData(reset: true);
          Navigator.pop(c);
        },
      ),
    );
  }

  void _showForm({int? roleId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _RoleFormSheet(
        roleId: roleId,
        onSaved: () {
          _fetchData(reset: true);
          Navigator.pop(c);
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:role:*')) {
      return Scaffold(
        appBar: AppBar(title: const Text('角色管理')),
        body: const Center(child: Text('无权限访问')),
      );
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('角色管理'),
        actions: [
          if (auth.hasPerm('sys:role:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增角色',
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
                hintText: '搜索角色名称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    _fetchData(reset: true);
                  },
                ),
              ),
              onSubmitted: (_) => _fetchData(reset: true),
            ),
          ),
          Expanded(child: _buildList(theme)),
        ],
      ),
    );
  }

  Widget _buildList(ThemeData theme) {
    if (_loading && _items.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _error!,
              style: TextStyle(color: theme.colorScheme.error),
            ),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => _fetchData(reset: true),
              child: const Text('重试'),
            ),
          ],
        ),
      );
    }
    if (_items.isEmpty) {
      return const Center(child: Text('暂无数据'));
    }

    return RefreshIndicator(
      onRefresh: () => _fetchData(reset: true),
      child: ListView.builder(
        itemCount: _items.length + (_items.length < _total ? 1 : 0),
        itemBuilder: (context, index) {
          if (index >= _items.length) {
            if (!_loading) {
              _pageNum++;
              _fetchData();
            }
            return const Center(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              ),
            );
          }
          final item = _items[index];
          return Card(
            child: ListTile(
              leading: CircleAvatar(
                child: Text(
                  ((item.name ?? 'R')[0]).toUpperCase(),
                ),
              ),
              title: Text(item.name ?? ''),
              subtitle: Text('编码: ${item.code ?? ''}'),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.security, size: 20),
                    tooltip: '权限分配',
                    onPressed: () => _showPermissionDialog(
                        item.id, item.name ?? ''),
                  ),
                  IconButton(
                    icon: const Icon(Icons.edit, size: 20),
                    tooltip: '编辑',
                    onPressed: () => _showForm(roleId: item.id),
                  ),
                  IconButton(
                    icon: Icon(Icons.delete,
                        size: 20, color: AppTheme.errorColor),
                    tooltip: '删除',
                    onPressed: () => _deleteRole(item.id),
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

class _RoleFormSheet extends ConsumerStatefulWidget {
  const _RoleFormSheet({this.roleId, required this.onSaved});
  final int? roleId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_RoleFormSheet> createState() => _RoleFormSheetState();
}

class _RoleFormSheetState extends ConsumerState<_RoleFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _codeCtrl = TextEditingController();
  final _sortCtrl = TextEditingController();
  int _status = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.roleId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _codeCtrl.dispose();
    _sortCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref.read(roleServiceProvider).getById(widget.roleId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name;
          _codeCtrl.text = data.code;
          _sortCtrl.text = (data.sort ?? 0).toString();
          _status = data.status ?? 1;
        });
      }
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    final form = RoleForm(
      name: _nameCtrl.text.trim(),
      code: _codeCtrl.text.trim(),
      sort: int.tryParse(_sortCtrl.text) ?? 0,
      status: _status,
    );
    try {
      final service = ref.read(roleServiceProvider);
      if (widget.roleId != null) {
        await service.update(widget.roleId!, form);
      } else {
        await service.add(form);
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
    final isNew = widget.roleId == null;
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
              Text(isNew ? '新增角色' : '编辑角色',
                  style: Theme.of(context).textTheme.titleLarge),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '角色名称'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _codeCtrl,
                decoration: const InputDecoration(labelText: '角色编码'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _sortCtrl,
                decoration: const InputDecoration(labelText: '排序'),
                keyboardType: TextInputType.number,
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

class _RolePermissionSheet extends ConsumerStatefulWidget {
  const _RolePermissionSheet({
    required this.roleId,
    required this.roleName,
    required this.onSaved,
  });
  final int roleId;
  final String roleName;
  final VoidCallback onSaved;
  @override
  ConsumerState<_RolePermissionSheet> createState() =>
      _RolePermissionSheetState();
}

class _RolePermissionSheetState extends ConsumerState<_RolePermissionSheet> {
  List<Menu> _menus = [];
  Set<int> _checkedIds = {};
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    try {
      final menus = await ref.read(menuServiceProvider).getList();
      final menuIds =
          await ref.read(roleServiceProvider).getRoleMenuIds(widget.roleId);
      if (mounted) {
        setState(() {
          _menus = menus;
          _checkedIds = menuIds.toSet();
          _loading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _save() async {
    setState(() => _loading = true);
    try {
      await ref
          .read(roleServiceProvider)
          .updateRoleMenus(widget.roleId, _checkedIds.toList());
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

  List<Menu> _flattenTree(List<Menu> list) {
    final result = <Menu>[];
    for (final item in list) {
      result.add(item);
      final children = item.children;
      if (children != null && children.isNotEmpty) {
        result.addAll(_flattenTree(children));
      }
    }
    return result;
  }

  @override
  Widget build(BuildContext context) {
    final flat = _flattenTree(_menus);
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: EdgeInsets.all(AppTheme.spacingM),
            child: Row(
              children: [
                Text('分配权限 - ${widget.roleName}',
                    style: Theme.of(context).textTheme.titleMedium),
                const Spacer(),
                TextButton(
                  onPressed: _loading ? null : _save,
                  child: const Text('保存'),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          SizedBox(
            height: 400,
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : ListView(
                    children: flat.map((menu) {
                      final depth =
                          menu.parentId != 0 ? 1 : 0;
                      return CheckboxListTile(
                        title: Text(menu.name),
                        contentPadding:
                            EdgeInsets.only(left: 16.0 + depth * 24.0),
                        value: _checkedIds.contains(menu.id),
                        onChanged: (checked) {
                          setState(() {
                            if (checked == true) {
                              _checkedIds.add(menu.id);
                            } else {
                              _checkedIds.remove(menu.id);
                            }
                          });
                        },
                      );
                    }).toList(),
                  ),
          ),
        ],
      ),
    );
  }
}
