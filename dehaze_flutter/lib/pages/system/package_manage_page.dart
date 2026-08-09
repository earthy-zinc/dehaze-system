import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/package_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/package_service.dart';
import '../../theme/app_theme.dart';

final packageManageProvider = StateNotifierProvider<
    PackageManageNotifier, AsyncValue<PagedList<PackagePageVO>>>(
  (ref) => PackageManageNotifier(ref.watch(packageServiceProvider)),
);

class PackageManageNotifier extends PagedListNotifier<PackagePageVO> {
  PackageManageNotifier(this._service) : super();
  final PackageService _service;

  @override
  Future<PageResult<PackagePageVO>> fetchPage(int pageNum) {
    return _service.getPage(
      PackageQuery(pageNum: pageNum, pageSize: pageSize, keyword: keyword),
    );
  }
}

/// 套餐管理页面（L2）
///
/// 权限：sys:package:*
class PackageManagePage extends ConsumerStatefulWidget {
  const PackageManagePage({super.key});

  @override
  ConsumerState<PackageManagePage> createState() => _PackageManagePageState();
}

class _PackageManagePageState extends ConsumerState<PackageManagePage> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _toggleStatus(int id, int currentStatus) async {
    final newStatus = currentStatus == 1 ? 0 : 1;
    try {
      await ref.read(packageServiceProvider).updateStatus(id, newStatus);
      _showSnack('状态已更新');
      ref.read(packageManageProvider.notifier).refresh();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _delete(int id) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('确认删除'),
        content: const Text('确定要删除该套餐吗？'),
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
      await ref.read(packageServiceProvider).delete(id);
      _showSnack('删除成功');
      ref.read(packageManageProvider.notifier).refresh();
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

  void _showForm({int? packageId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _PackageFormSheet(
        packageId: packageId,
        onSaved: () {
          ref.read(packageManageProvider.notifier).refresh();
          Navigator.pop(c);
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    final theme = Theme.of(context);
    final state = ref.watch(packageManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('套餐管理'),
        actions: [
          if (auth.hasPerm('sys:package:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增套餐',
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
                hintText: '搜索套餐名称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    ref.read(packageManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) => ref.read(packageManageProvider.notifier).search(v),
            ),
          ),
          Expanded(child: _buildBody(theme, state)),
        ],
      ),
    );
  }

  Widget _buildBody(ThemeData theme, AsyncValue<PagedList<PackagePageVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(packageManageProvider.notifier).refresh(),
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
          onRefresh: () => ref.read(packageManageProvider.notifier).refresh(),
          child: LoadMoreListener(
            onLoadMore: () => ref.read(packageManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                final status = item.status;
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.shopping_bag),
                    title: Text(item.name),
                    subtitle: Text('¥${item.currentPrice} | ${item.periodName}'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: Icon(
                            status == 1 ? Icons.visibility_off : Icons.visibility,
                            size: 20,
                          ),
                          tooltip: status == 1 ? '下架' : '上架',
                          onPressed: () => _toggleStatus(item.id, status),
                        ),
                        IconButton(
                          icon: const Icon(Icons.edit, size: 20),
                          onPressed: () => _showForm(packageId: item.id),
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

class _PackageFormSheet extends ConsumerStatefulWidget {
  const _PackageFormSheet({this.packageId, required this.onSaved});

  final int? packageId;
  final VoidCallback onSaved;

  @override
  ConsumerState<_PackageFormSheet> createState() => _PackageFormSheetState();
}

class _PackageFormSheetState extends ConsumerState<_PackageFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _priceCtrl = TextEditingController();
  final _durationCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    if (widget.packageId != null) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _priceCtrl.dispose();
    _durationCtrl.dispose();
    _descCtrl.dispose();
    super.dispose();
  }

  PackageService get _service => ref.read(packageServiceProvider);

  Future<void> _loadData() async {
    try {
      final data = await _service.getById(widget.packageId!);
      if (!mounted) {
        return;
      }
      setState(() {
        _nameCtrl.text = data.name;
        _priceCtrl.text = data.currentPrice.toString();
        _durationCtrl.text = data.period;
        _descCtrl.text = data.description ?? '';
      });
    } catch (_) {}
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    try {
      final form = PackageForm(
        name: _nameCtrl.text.trim(),
        level: 'standard',
        period: _durationCtrl.text.trim(),
        originalPrice: double.tryParse(_priceCtrl.text) ?? 0,
        currentPrice: double.tryParse(_priceCtrl.text) ?? 0,
        description: _descCtrl.text.isNotEmpty ? _descCtrl.text.trim() : null,
      );
      if (widget.packageId != null) {
        await _service.update(widget.packageId!, form);
      } else {
        await _service.add(form);
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
    final isNew = widget.packageId == null;
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
                isNew ? '新增套餐' : '编辑套餐',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '套餐名称'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _priceCtrl,
                decoration: const InputDecoration(labelText: '价格'),
                keyboardType: TextInputType.number,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _durationCtrl,
                decoration: const InputDecoration(labelText: '有效天数'),
                keyboardType: TextInputType.number,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '必填' : null,
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
