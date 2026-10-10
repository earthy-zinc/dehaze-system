import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/dict_model.dart';
import '../../providers/providers.dart';
import '../../services/dict_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/ui_utils.dart';

final dictItemManageProvider =
    StateNotifierProvider.family<
      DictItemManageNotifier,
      AsyncValue<PagedList<Dict>>,
      String
    >(
      (ref, typeCode) =>
          DictItemManageNotifier(ref.watch(dictServiceProvider), typeCode),
    );

class DictItemManageNotifier extends PagedListNotifier<Dict> {
  DictItemManageNotifier(this._service, this.typeCode) : super(pageSize: 20);
  final DictService _service;
  final String typeCode;

  @override
  Future<PageResult<Dict>> fetchPage(int pageNum) {
    return _service.getDictPage(
      DictQuery(pageNum: pageNum, pageSize: 20, typeCode: typeCode),
    );
  }
}

/// 字典项管理页面
class DictItemManagePage extends ConsumerWidget {
  const DictItemManagePage({
    super.key,
    required this.typeCode,
    required this.typeName,
  });
  final String typeCode;
  final String typeName;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final state = ref.watch(dictItemManageProvider(typeCode));

    Future<void> deleteItem(int id) async {
      try {
        await ref.read(dictServiceProvider).deleteDict(id);
        ref.read(dictItemManageProvider(typeCode).notifier).refresh();
        if (!context.mounted) return;
        showSnackBar(context, '删除成功');
      } catch (e) {
        if (!context.mounted) return;
        showSnackBar(context, extractErrorMessage(e));
      }
    }

    void showForm({int? itemId}) {
      showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        useSafeArea: true,
        builder: (c) => _DictItemFormSheet(
          typeCode: typeCode,
          itemId: itemId,
          onSaved: () {
            ref.read(dictItemManageProvider(typeCode).notifier).refresh();
            Navigator.pop(c);
          },
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(typeName),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            tooltip: '新增字典项',
            onPressed: () => showForm(),
          ),
        ],
      ),
      body: state.when(
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
                onPressed: () => ref
                    .read(dictItemManageProvider(typeCode).notifier)
                    .refresh(),
                child: const Text('重试'),
              ),
            ],
          ),
        ),
        data: (page) => page.items.isEmpty
            ? const Center(child: Text('暂无字典项'))
            : RefreshIndicator(
                onRefresh: () => ref
                    .read(dictItemManageProvider(typeCode).notifier)
                    .refresh(),
                child: LoadMoreListener(
                  onLoadMore: () => ref
                      .read(dictItemManageProvider(typeCode).notifier)
                      .loadMore(),
                  child: ListView.builder(
                    itemCount: page.items.length,
                    itemBuilder: (context, index) {
                      final item = page.items[index];
                      return Card(
                        child: ListTile(
                          title: Text(item.label ?? ''),
                          subtitle: Text('值: ${item.value} | 排序: ${item.sort}'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                icon: const Icon(Icons.edit, size: 20),
                                onPressed: () => showForm(itemId: item.id),
                              ),
                              IconButton(
                                icon: Icon(
                                  Icons.delete,
                                  size: 20,
                                  color: AppTheme.errorColor,
                                ),
                                onPressed: () => deleteItem(item.id!),
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
    );
  }
}

class _DictItemFormSheet extends ConsumerStatefulWidget {
  const _DictItemFormSheet({
    required this.typeCode,
    this.itemId,
    required this.onSaved,
  });
  final String typeCode;
  final int? itemId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_DictItemFormSheet> createState() => _DictItemFormSheetState();
}

class _DictItemFormSheetState extends ConsumerState<_DictItemFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _valueCtrl = TextEditingController();
  final _sortCtrl = TextEditingController();
  int _status = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.itemId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _valueCtrl.dispose();
    _sortCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref
          .read(dictServiceProvider)
          .getDictForm(widget.itemId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.label ?? '';
          _valueCtrl.text = data.value ?? '';
          _sortCtrl.text = (data.sort ?? 1).toString();
          _status = data.status ?? 1;
        });
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('字典数据信息加载失败')));
      }
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    try {
      final form = DictForm(
        label: _nameCtrl.text.trim(),
        value: _valueCtrl.text.trim(),
        sort: int.tryParse(_sortCtrl.text) ?? 1,
        status: _status,
        typeCode: widget.typeCode,
      );
      if (widget.itemId != null) {
        await ref.read(dictServiceProvider).updateDict(widget.itemId!, form);
      } else {
        await ref.read(dictServiceProvider).addDict(form);
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
    final isNew = widget.itemId == null;
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
                isNew ? '新增字典项' : '编辑字典项',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '字典名称'),
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _valueCtrl,
                decoration: const InputDecoration(labelText: '字典值'),
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
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
