import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/dataset_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/dataset_service.dart';
import '../../theme/app_theme.dart';

final datasetManageProvider =
    StateNotifierProvider<DatasetManageNotifier, AsyncValue<PagedList<Dataset>>>(
  (ref) => DatasetManageNotifier(ref.watch(datasetServiceProvider)),
);

class DatasetManageNotifier extends PagedListNotifier<Dataset> {
  DatasetManageNotifier(this._service) : super();
  final DatasetService _service;

  @override
  Future<PageResult<Dataset>> fetchPage(int pageNum) {
    return _service.getList(
      DatasetQuery(pageNum: pageNum, pageSize: pageSize, keyword: keyword),
    );
  }
}

/// 数据集管理页面（L2）
///
/// 权限：sys:dataset:*
class DatasetManagePage extends ConsumerStatefulWidget {
  const DatasetManagePage({super.key});

  @override
  ConsumerState<DatasetManagePage> createState() => _DatasetManagePageState();
}

class _DatasetManagePageState extends ConsumerState<DatasetManagePage> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _delete(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder:
          (c) => AlertDialog(
            title: const Text('确认删除'),
            content: const Text('确定要删除该数据集吗？'),
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
      await ref.read(datasetServiceProvider).deleteById(id);
      _showSnack('删除成功');
      ref.read(datasetManageProvider.notifier).refresh();
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

  void _showForm({int? datasetId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder:
          (c) => _DatasetFormSheet(
            datasetId: datasetId,
            onSaved: () {
              ref.read(datasetManageProvider.notifier).refresh();
              Navigator.pop(c);
            },
          ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    final theme = Theme.of(context);
    final state = ref.watch(datasetManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('数据集管理'),
        actions: [
          if (auth.hasPerm('sys:dataset:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增数据集',
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
                hintText: '搜索数据集名称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    ref.read(datasetManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) => ref.read(datasetManageProvider.notifier).search(v),
            ),
          ),
          Expanded(child: _buildBody(theme, state)),
        ],
      ),
    );
  }

  Widget _buildBody(ThemeData theme, AsyncValue<PagedList<Dataset>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(datasetManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) {
        if (page.items.isEmpty) {
          return const Center(child: Text('暂无数据'));
        }
        return RefreshIndicator(
          onRefresh: () => ref.read(datasetManageProvider.notifier).refresh(),
          child: LoadMoreListener(
            onLoadMore: () => ref.read(datasetManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.storage),
                    title: Text(item.name),
                    subtitle: Text(
                      '${item.statistics?.itemCount ?? 0} 张图片',
                    ),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.edit, size: 20),
                          onPressed: () => _showForm(datasetId: item.id),
                        ),
                        IconButton(
                          icon: Icon(Icons.delete, size: 20, color: AppTheme.errorColor),
                          onPressed: () => _delete(item.id),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        );
      },
    );
  }
}

class _DatasetFormSheet extends ConsumerStatefulWidget {
  const _DatasetFormSheet({this.datasetId, required this.onSaved});
  final int? datasetId;
  final VoidCallback onSaved;
  @override
  ConsumerState<_DatasetFormSheet> createState() => _DatasetFormSheetState();
}

class _DatasetFormSheetState extends ConsumerState<_DatasetFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  int _type = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.datasetId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _descCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadData() async {
    try {
      final data = await ref
          .read(datasetServiceProvider)
          .getDatasetInfoById(widget.datasetId!);
      if (mounted) {
        setState(() {
          _nameCtrl.text = data.name;
          _descCtrl.text = data.description ?? '';
          _type = int.tryParse(data.type) ?? 1;
        });
      }
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    try {
      final service = ref.read(datasetServiceProvider);
      if (widget.datasetId != null) {
        await service.update(
          widget.datasetId!,
          DatasetUpdateForm(
            name: _nameCtrl.text.trim(),
            type: _type.toString(),
            description:
                _descCtrl.text.isNotEmpty ? _descCtrl.text.trim() : null,
          ),
        );
      } else {
        await service.add(
          DatasetAddForm(
            parentId: 0,
            name: _nameCtrl.text.trim(),
            type: _type.toString(),
            description:
                _descCtrl.text.isNotEmpty ? _descCtrl.text.trim() : null,
          ),
        );
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
    final isNew = widget.datasetId == null;
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
                isNew ? '新增数据集' : '编辑数据集',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '数据集名称'),
                validator:
                    (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _descCtrl,
                decoration: const InputDecoration(labelText: '描述'),
                maxLines: 3,
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
