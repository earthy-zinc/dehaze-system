import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/dict_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/dict_service.dart';
import '../../theme/app_theme.dart';
import 'dict_item_page.dart';

final dictTypeManageProvider =
    StateNotifierProvider<
      DictTypeManageNotifier,
      AsyncValue<PagedList<DictType>>
    >((ref) => DictTypeManageNotifier(ref.watch(dictServiceProvider)));

class DictTypeManageNotifier extends PagedListNotifier<DictType> {
  DictTypeManageNotifier(this._service) : super();
  final DictService _service;

  @override
  Future<PageResult<DictType>> fetchPage(int pageNum) {
    return _service.getTypePage(
      DictTypeQuery(pageNum: pageNum, pageSize: 10, keywords: keyword),
    );
  }
}

/// 字典管理页面（L2）
///
/// 权限：sys:dict:*
class DictManagePage extends ConsumerStatefulWidget {
  const DictManagePage({super.key});

  @override
  ConsumerState<DictManagePage> createState() => _DictManagePageState();
}

class _DictManagePageState extends ConsumerState<DictManagePage> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _deleteType(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('确认删除'),
        content: const Text('确定要删除该字典类型吗？'),
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
      await ref.read(dictServiceProvider).deleteType(id);
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('删除成功')));
      }
      ref.read(dictTypeManageProvider.notifier).refresh();
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

  void _showTypeForm({int? typeId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _DictTypeFormSheet(
        typeId: typeId,
        onSaved: () {
          ref.read(dictTypeManageProvider.notifier).refresh();
          Navigator.pop(c);
        },
      ),
    );
  }

  void _showDictItems(String typeCode, String typeName) {
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) =>
            DictItemManagePage(typeCode: typeCode, typeName: typeName),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    final theme = Theme.of(context);
    final state = ref.watch(dictTypeManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('字典管理'),
        actions: [
          if (auth.hasPerm('sys:dict:type:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增字典类型',
              onPressed: () => _showTypeForm(),
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
                hintText: '搜索字典类型',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    ref.read(dictTypeManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) =>
                  ref.read(dictTypeManageProvider.notifier).search(v),
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
                          ref.read(dictTypeManageProvider.notifier).refresh(),
                      child: const Text('重试'),
                    ),
                  ],
                ),
              ),
              data: (page) => page.items.isEmpty
                  ? const Center(child: Text('暂无数据'))
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(dictTypeManageProvider.notifier).refresh(),
                      child: LoadMoreListener(
                        onLoadMore: () => ref
                            .read(dictTypeManageProvider.notifier)
                            .loadMore(),
                        child: ListView.builder(
                          itemCount: page.items.length,
                          itemBuilder: (context, index) {
                            final item = page.items[index];
                            return Card(
                              child: ListTile(
                                leading: const Icon(Icons.list_alt),
                                title: Text(item.name),
                                subtitle: Text('编码: ${item.code}'),
                                trailing: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    IconButton(
                                      icon: const Icon(
                                        Icons.format_list_bulleted,
                                        size: 20,
                                      ),
                                      tooltip: '管理字典项',
                                      onPressed: () =>
                                          _showDictItems(item.code, item.name),
                                    ),
                                    IconButton(
                                      icon: const Icon(Icons.edit, size: 20),
                                      tooltip: '编辑',
                                      onPressed: () =>
                                          _showTypeForm(typeId: item.id),
                                    ),
                                    IconButton(
                                      icon: Icon(
                                        Icons.delete,
                                        size: 20,
                                        color: AppTheme.errorColor,
                                      ),
                                      tooltip: '删除',
                                      onPressed: () => _deleteType(item.id),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
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

class _DictTypeFormSheet extends ConsumerStatefulWidget {
  const _DictTypeFormSheet({this.typeId, required this.onSaved});
  final int? typeId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_DictTypeFormSheet> createState() => _DictTypeFormSheetState();
}

class _DictTypeFormSheetState extends ConsumerState<_DictTypeFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _codeCtrl = TextEditingController();
  final _remarkCtrl = TextEditingController();
  int _status = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.typeId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _codeCtrl.dispose();
    _remarkCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref
          .read(dictServiceProvider)
          .getTypeForm(widget.typeId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name ?? '';
          _codeCtrl.text = data.code ?? '';
          _remarkCtrl.text = data.remark ?? '';
          _status = data.status;
        });
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('字典类型信息加载失败')));
      }
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    try {
      final form = DictTypeForm(
        name: _nameCtrl.text.trim(),
        code: _codeCtrl.text.trim(),
        status: _status,
        remark: _remarkCtrl.text.trim(),
      );
      if (widget.typeId != null) {
        await ref.read(dictServiceProvider).updateType(widget.typeId!, form);
      } else {
        await ref.read(dictServiceProvider).addType(form);
      }
      widget.onSaved();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(extractErrorMessage(e))));
      }
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isNew = widget.typeId == null;
    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      child: SingleChildScrollView(
        padding: EdgeInsets.all(AppTheme.spacingL),
        child: Form(
          key: _formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                isNew ? '新增字典类型' : '编辑字典类型',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '类型名称'),
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _codeCtrl,
                decoration: const InputDecoration(labelText: '类型编码'),
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _remarkCtrl,
                decoration: const InputDecoration(labelText: '备注'),
                maxLines: 2,
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
