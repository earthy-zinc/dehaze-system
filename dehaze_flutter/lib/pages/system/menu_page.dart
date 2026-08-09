import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/menu_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/menu_service.dart';
import '../../theme/app_theme.dart';

final menuManageProvider =
    StateNotifierProvider<MenuManageNotifier, AsyncValue<PagedList<Menu>>>(
  (ref) => MenuManageNotifier(ref.watch(menuServiceProvider)),
);

class MenuManageNotifier extends PagedListNotifier<Menu> {
  MenuManageNotifier(this._service) : super();
  final MenuService _service;

  @override
  Future<PageResult<Menu>> fetchPage(int pageNum) async {
    final list = await _service.getList(name: keyword);
    return PageResult(list: list, total: list.length);
  }
}

/// 菜单管理页面（L2）
///
/// 权限：sys:menu:*
class MenuManagePage extends ConsumerStatefulWidget {
  const MenuManagePage({super.key});

  @override
  ConsumerState<MenuManagePage> createState() => _MenuManagePageState();
}

class _MenuManagePageState extends ConsumerState<MenuManagePage> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _deleteMenu(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('确认删除'),
        content: const Text('删除菜单将同时删除子菜单，确定？'),
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
      await ref.read(menuServiceProvider).deleteByIds([id]);
      _showSnack('删除成功');
      ref.read(menuManageProvider.notifier).refresh();
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

  void _showForm({int? menuId, int? parentId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _MenuFormSheet(
        menuId: menuId,
        parentId: parentId,
        onSaved: () {
          ref.read(menuManageProvider.notifier).refresh();
          Navigator.pop(c);
        },
      ),
    );
  }

  List<Widget> _buildTree(List<Menu> items, int depth) {
    final widgets = <Widget>[];
    for (final item in items) {
      final children = item.children;
      widgets.add(
        Card(
          child: ListTile(
            contentPadding:
                EdgeInsets.only(left: 16.0 + depth * 24.0, right: 8),
            leading: Icon(
              item.type == 2 ? Icons.smart_button : Icons.folder_outlined,
              size: 20,
            ),
            title: Text(
              item.name,
              style: TextStyle(
                fontWeight: depth == 0 ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
            subtitle: Text(item.path ?? ''),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (item.type == 1)
                  IconButton(
                    icon: const Icon(Icons.add, size: 20),
                    tooltip: '添加子菜单',
                    onPressed: () => _showForm(parentId: item.id),
                  ),
                IconButton(
                  icon: const Icon(Icons.edit, size: 20),
                  tooltip: '编辑',
                  onPressed: () => _showForm(menuId: item.id),
                ),
                IconButton(
                  icon:
                      Icon(Icons.delete, size: 20, color: AppTheme.errorColor),
                  tooltip: '删除',
                  onPressed: () => _deleteMenu(item.id),
                ),
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
    final theme = Theme.of(context);
    final state = ref.watch(menuManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('菜单管理'),
        actions: [
          if (auth.hasPerm('sys:menu:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增顶级菜单',
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
                hintText: '搜索菜单名称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    ref.read(menuManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) => ref.read(menuManageProvider.notifier).search(v),
            ),
          ),
          Expanded(
            child: state.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      extractErrorMessage(e),
                      style: TextStyle(color: theme.colorScheme.error),
                    ),
                    SizedBox(height: AppTheme.spacingM),
                    FilledButton(
                      onPressed: () =>
                          ref.read(menuManageProvider.notifier).refresh(),
                      child: const Text('重试'),
                    ),
                  ],
                ),
              ),
              data: (page) => page.items.isEmpty
                  ? const Center(child: Text('暂无数据'))
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(menuManageProvider.notifier).refresh(),
                      child: LoadMoreListener(
                        onLoadMore: () =>
                            ref.read(menuManageProvider.notifier).loadMore(),
                        child: ListView(
                          children: _buildTree(page.items, 0),
                        ),
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MenuFormSheet extends ConsumerStatefulWidget {
  const _MenuFormSheet({this.menuId, this.parentId, required this.onSaved});
  final int? menuId;
  final int? parentId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_MenuFormSheet> createState() => _MenuFormSheetState();
}

class _MenuFormSheetState extends ConsumerState<_MenuFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _pathCtrl = TextEditingController();
  final _componentCtrl = TextEditingController();
  final _iconCtrl = TextEditingController();
  final _sortCtrl = TextEditingController();
  int _type = 1;
  int _visible = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.menuId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _pathCtrl.dispose();
    _componentCtrl.dispose();
    _iconCtrl.dispose();
    _sortCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref.read(menuServiceProvider).getById(widget.menuId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name;
          _pathCtrl.text = data.path ?? '';
          _componentCtrl.text = data.component ?? '';
          _iconCtrl.text = data.icon ?? '';
          _sortCtrl.text = data.sort.toString();
          _type = data.type;
          _visible = data.visible;
        });
      }
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    final form = MenuForm(
      parentId: widget.parentId ?? 0,
      name: _nameCtrl.text.trim(),
      type: _type,
      visible: _visible,
      sort: int.tryParse(_sortCtrl.text) ?? 0,
      status: 1,
      path: _pathCtrl.text.trim().isEmpty ? null : _pathCtrl.text.trim(),
      component: _componentCtrl.text.trim().isEmpty
          ? null
          : _componentCtrl.text.trim(),
      icon: _iconCtrl.text.trim().isEmpty ? null : _iconCtrl.text.trim(),
    );
    try {
      final service = ref.read(menuServiceProvider);
      if (widget.menuId != null) {
        await service.update(widget.menuId!, form);
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
    final isNew = widget.menuId == null;
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
              Text(isNew ? '新增菜单' : '编辑菜单',
                  style: Theme.of(context).textTheme.titleLarge),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '菜单名称'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _pathCtrl,
                decoration: const InputDecoration(labelText: '路由路径'),
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _componentCtrl,
                decoration: const InputDecoration(labelText: '组件路径'),
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _iconCtrl,
                decoration: const InputDecoration(labelText: '图标'),
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _sortCtrl,
                decoration: const InputDecoration(labelText: '排序'),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: AppTheme.spacingM),
              DropdownButtonFormField<int>(
                initialValue: _type,
                decoration: const InputDecoration(labelText: '类型'),
                items: const [
                  DropdownMenuItem(value: 1, child: Text('目录')),
                  DropdownMenuItem(value: 2, child: Text('菜单')),
                  DropdownMenuItem(value: 3, child: Text('按钮')),
                ],
                onChanged: (v) => _type = v ?? 1,
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
