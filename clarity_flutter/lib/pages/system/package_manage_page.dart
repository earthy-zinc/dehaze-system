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
import '../../utils/format_utils.dart';

final packageManageProvider =
    StateNotifierProvider<
      PackageManageNotifier,
      AsyncValue<PagedList<PackagePageVO>>
    >((ref) => PackageManageNotifier(ref.watch(packageServiceProvider)));

class PackageManageNotifier extends PagedListNotifier<PackagePageVO> {
  PackageManageNotifier(this._service) : super();
  final PackageService _service;

  @override
  Future<PageResult<PackagePageVO>> fetchPage(int pageNum) {
    return _service.getPage(
      PackageQuery(pageNum: pageNum, pageSize: pageSize, name: keyword),
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
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('状态已更新')));
      }
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
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('删除成功')));
      }
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

  String _spec(PackagePageVO item) => item.packageType == 'credit'
      ? '${item.creditAmount ?? 0} 积分'
      : (packagePeriodLabels[item.period] ?? item.period ?? '');

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
          if (auth.hasPerm('package:add'))
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
              onSubmitted: (v) =>
                  ref.read(packageManageProvider.notifier).search(v),
            ),
          ),
          Expanded(child: _buildBody(theme, state)),
        ],
      ),
    );
  }

  Widget _buildBody(
    ThemeData theme,
    AsyncValue<PagedList<PackagePageVO>> state,
  ) {
    return state.when(
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
                  ref.read(packageManageProvider.notifier).refresh(),
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
            onLoadMore: () =>
                ref.read(packageManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                final status = item.status;
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.shopping_bag),
                    title: Text(item.name),
                    subtitle: Text(
                      '${formatYuan(item.salePrice)} | ${_spec(item)}',
                    ),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: Icon(
                            status == 1
                                ? Icons.visibility_off
                                : Icons.visibility,
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
                          icon: Icon(
                            Icons.delete,
                            size: 20,
                            color: AppTheme.errorColor,
                          ),
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
  final _originalPriceCtrl = TextEditingController();
  final _salePriceCtrl = TextEditingController();
  final _periodDaysCtrl = TextEditingController();
  final _creditAmountCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  String _packageType = 'vip';
  String _levelCode = 'level_1';
  String _period = 'monthly';
  bool _loading = false;

  bool get _isEdit => widget.packageId != null;
  bool get _isVip => _packageType == 'vip';

  @override
  void initState() {
    super.initState();
    if (_isEdit) {
      _loadData();
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _originalPriceCtrl.dispose();
    _salePriceCtrl.dispose();
    _periodDaysCtrl.dispose();
    _creditAmountCtrl.dispose();
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
        _packageType = data.packageType;
        _levelCode = data.levelCode ?? _levelCode;
        _period = data.period ?? _period;
        _periodDaysCtrl.text = data.periodDays?.toString() ?? '';
        _creditAmountCtrl.text = data.creditAmount?.toString() ?? '';
        _originalPriceCtrl.text = _fenToYuan(data.originalPrice);
        _salePriceCtrl.text = _fenToYuan(data.salePrice);
        _descCtrl.text = data.description ?? '';
      });
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('套餐信息加载失败')));
      }
    }
  }

  String _fenToYuan(int fen) => (fen / 100).toStringAsFixed(2);

  int _yuanToFen(String text) =>
      ((double.tryParse(text.trim()) ?? 0) * 100).round();

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() => _loading = true);
    try {
      final form = PackageForm(
        name: _nameCtrl.text.trim(),
        packageType: _packageType,
        originalPrice: _yuanToFen(_originalPriceCtrl.text),
        salePrice: _yuanToFen(_salePriceCtrl.text),
        levelCode: _isVip ? _levelCode : null,
        period: _isVip ? _period : null,
        periodDays: _isVip ? int.tryParse(_periodDaysCtrl.text.trim()) : null,
        creditAmount: _isVip
            ? null
            : int.tryParse(_creditAmountCtrl.text.trim()),
        description: _descCtrl.text.isNotEmpty ? _descCtrl.text.trim() : null,
      );
      if (_isEdit) {
        await _service.update(widget.packageId!, form);
      } else {
        await _service.add(form);
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
                _isEdit ? '编辑套餐' : '新增套餐',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(labelText: '套餐名称'),
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              DropdownButtonFormField<String>(
                initialValue: _packageType,
                decoration: const InputDecoration(labelText: '商品类型'),
                // 商品类型创建后不可修改，编辑时禁用
                onChanged: _isEdit
                    ? null
                    : (v) => setState(() => _packageType = v ?? 'vip'),
                items: const [
                  DropdownMenuItem(value: 'vip', child: Text('会员卡')),
                  DropdownMenuItem(value: 'credit', child: Text('积分卡')),
                ],
              ),
              SizedBox(height: AppTheme.spacingM),
              if (_isVip) ...[
                DropdownButtonFormField<String>(
                  initialValue: _levelCode,
                  decoration: const InputDecoration(labelText: '会员等级'),
                  onChanged: (v) => setState(() => _levelCode = v ?? 'level_1'),
                  items: const [
                    DropdownMenuItem(value: 'level_1', child: Text('等级一')),
                    DropdownMenuItem(value: 'level_2', child: Text('等级二')),
                    DropdownMenuItem(value: 'level_3', child: Text('等级三')),
                  ],
                ),
                SizedBox(height: AppTheme.spacingM),
                DropdownButtonFormField<String>(
                  initialValue: _period,
                  decoration: const InputDecoration(labelText: '计费周期'),
                  onChanged: (v) => setState(() => _period = v ?? 'monthly'),
                  items: packagePeriodLabels.entries
                      .map(
                        (e) => DropdownMenuItem(
                          value: e.key,
                          child: Text(e.value),
                        ),
                      )
                      .toList(),
                ),
                SizedBox(height: AppTheme.spacingM),
                TextFormField(
                  controller: _periodDaysCtrl,
                  decoration: const InputDecoration(labelText: '有效期天数'),
                  keyboardType: TextInputType.number,
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? '必填' : null,
                ),
              ] else ...[
                TextFormField(
                  controller: _creditAmountCtrl,
                  decoration: const InputDecoration(labelText: '可得积分'),
                  keyboardType: TextInputType.number,
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? '必填' : null,
                ),
              ],
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _originalPriceCtrl,
                decoration: const InputDecoration(labelText: '原价（元）'),
                keyboardType: TextInputType.number,
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _salePriceCtrl,
                decoration: const InputDecoration(labelText: '促销价（元）'),
                keyboardType: TextInputType.number,
                validator: (v) => (v == null || v.trim().isEmpty) ? '必填' : null,
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
