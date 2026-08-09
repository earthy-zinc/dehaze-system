import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/dept_model.dart';
import '../../models/user_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/user_service.dart';
import '../../theme/app_theme.dart';

final userManageProvider =
    StateNotifierProvider<UserManageNotifier, AsyncValue<PagedList<UserPageVO>>>(
  (ref) => UserManageNotifier(ref.watch(userServiceProvider)),
);

class UserManageNotifier extends PagedListNotifier<UserPageVO> {
  UserManageNotifier(this._service) : super();
  final UserService _service;

  @override
  Future<PageResult<UserPageVO>> fetchPage(int pageNum) {
    return _service.getPage(
      UserQuery(pageNum: pageNum, pageSize: 10, keywords: keyword),
    );
  }
}

/// 用户管理页面（L2）
///
/// 权限：sys:user:*
class UserManagePage extends ConsumerStatefulWidget {
  const UserManagePage({super.key});

  @override
  ConsumerState<UserManagePage> createState() => _UserManagePageState();
}

class _UserManagePageState extends ConsumerState<UserManagePage> {
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _deleteUser(int userId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('确认删除'),
        content: const Text('确定要删除该用户吗？'),
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
      await ref.read(userServiceProvider).deleteByIds([userId]);
      _showSnack('删除成功');
      ref.read(userManageProvider.notifier).refresh();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _resetPassword(int userId) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('重置密码'),
        content: const Text('确定要重置密码为默认密码吗？'),
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
      await ref.read(userServiceProvider).updatePassword(userId, '123456');
      _showSnack('密码已重置');
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _toggleStatus(int userId, int currentStatus) async {
    try {
      await ref
          .read(userServiceProvider)
          .updateStatus(userId, currentStatus == 1 ? 0 : 1);
      _showSnack('状态已更新');
      ref.read(userManageProvider.notifier).refresh();
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

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    final theme = Theme.of(context);
    final state = ref.watch(userManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('用户管理'),
        actions: [
          if (auth.hasPerm('sys:user:add'))
            IconButton(
              icon: const Icon(Icons.add),
              tooltip: '新增用户',
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
                hintText: '搜索用户名/昵称',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    ref.read(userManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) => ref.read(userManageProvider.notifier).search(v),
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
                      style: theme.textTheme.bodyMedium
                          ?.copyWith(color: theme.colorScheme.error),
                    ),
                    SizedBox(height: AppTheme.spacingM),
                    FilledButton(
                      onPressed: () =>
                          ref.read(userManageProvider.notifier).refresh(),
                      child: const Text('重试'),
                    ),
                  ],
                ),
              ),
              data: (page) => page.items.isEmpty
                  ? const Center(child: Text('暂无数据'))
                  : RefreshIndicator(
                      onRefresh: () =>
                          ref.read(userManageProvider.notifier).refresh(),
                      child: LoadMoreListener(
                        onLoadMore: () =>
                            ref.read(userManageProvider.notifier).loadMore(),
                        child: ListView.builder(
                          itemCount: page.items.length,
                          itemBuilder: (context, index) {
                            final item = page.items[index];
                            final status = item.status ?? 1;
                            return Card(
                              child: ListTile(
                                leading: CircleAvatar(
                                  child: Text(
                                    ((item.nickname ??
                                                item.username ??
                                                '?')[0])
                                        .toUpperCase(),
                                  ),
                                ),
                                title: Text(item.nickname ?? item.username ?? ''),
                                subtitle: Text(item.username ?? ''),
                                trailing: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    IconButton(
                                      icon: Icon(Icons.lock_reset,
                                          size: 20, color: AppTheme.warningColor),
                                      tooltip: '重置密码',
                                      onPressed: () => _resetPassword(item.id),
                                    ),
                                    IconButton(
                                      icon: Icon(
                                        status == 1
                                            ? Icons.block
                                            : Icons.check_circle,
                                        size: 20,
                                        color: status == 1
                                            ? AppTheme.errorColor
                                            : AppTheme.successColor,
                                      ),
                                      tooltip: status == 1 ? '禁用' : '启用',
                                      onPressed: () =>
                                          _toggleStatus(item.id, status),
                                    ),
                                    IconButton(
                                      icon: const Icon(Icons.edit, size: 20),
                                      tooltip: '编辑',
                                      onPressed: () => _showForm(userId: item.id),
                                    ),
                                    IconButton(
                                      icon: Icon(Icons.delete,
                                          size: 20, color: AppTheme.errorColor),
                                      tooltip: '删除',
                                      onPressed: () => _deleteUser(item.id),
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

  void _showForm({int? userId}) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (c) => _UserFormSheet(
        userId: userId,
        onSaved: () {
          ref.read(userManageProvider.notifier).refresh();
          Navigator.pop(c);
        },
      ),
    );
  }
}

class _UserFormSheet extends ConsumerStatefulWidget {
  const _UserFormSheet({this.userId, required this.onSaved});
  final int? userId;
  final VoidCallback onSaved;

  @override
  ConsumerState<_UserFormSheet> createState() => _UserFormSheetState();
}

class _UserFormSheetState extends ConsumerState<_UserFormSheet> {
  final _formKey = GlobalKey<FormState>();
  final _usernameCtrl = TextEditingController();
  final _nicknameCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  int? _deptId;
  List<int> _roleIds = [];
  int _status = 1;
  bool _loading = false;
  List<DeptOption> _deptOptions = [];

  @override
  void initState() {
    super.initState();
    _loadOptions();
    if (widget.userId != null) {
      _loadFormData();
    }
  }

  @override
  void dispose() {
    _usernameCtrl.dispose();
    _nicknameCtrl.dispose();
    _passwordCtrl.dispose();
    _emailCtrl.dispose();
    _phoneCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadOptions() async {
    try {
      final deptService = ref.read(deptServiceProvider);
      final depts = await deptService.getOptions();
      if (mounted) {
        setState(() {
          _deptOptions = depts;
        });
      }
    } catch (_) {}
  }

  Future<void> _loadFormData() async {
    try {
      final data = await ref.read(userServiceProvider).getById(widget.userId!);
      if (mounted) {
        setState(() {
          _usernameCtrl.text = data.username;
          _nicknameCtrl.text = data.nickname ?? '';
          _emailCtrl.text = data.email ?? '';
          _phoneCtrl.text = data.phone ?? '';
          _deptId = data.deptId;
          _roleIds = data.roleIds;
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
    final form = UserForm(
      username: _usernameCtrl.text.trim(),
      nickname: _nicknameCtrl.text.trim(),
      email: _emailCtrl.text.trim().isEmpty ? null : _emailCtrl.text.trim(),
      phone: _phoneCtrl.text.trim().isEmpty ? null : _phoneCtrl.text.trim(),
      deptId: _deptId,
      roleIds: _roleIds,
      status: _status,
      password: widget.userId == null ? _passwordCtrl.text : null,
    );
    try {
      final service = ref.read(userServiceProvider);
      if (widget.userId != null) {
        await service.update(widget.userId!, form);
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
    final isNew = widget.userId == null;
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
              Text(isNew ? '新增用户' : '编辑用户',
                  style: Theme.of(context).textTheme.titleLarge),
              SizedBox(height: AppTheme.spacingL),
              TextFormField(
                controller: _usernameCtrl,
                decoration: const InputDecoration(labelText: '用户名'),
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? '请输入用户名' : null,
              ),
              SizedBox(height: AppTheme.spacingM),
              if (isNew)
                TextFormField(
                  controller: _passwordCtrl,
                  decoration: const InputDecoration(labelText: '密码'),
                  obscureText: true,
                  validator: (v) =>
                      (v == null || v.trim().isEmpty) ? '请输入密码' : null,
                ),
              if (isNew) SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _nicknameCtrl,
                decoration: const InputDecoration(labelText: '昵称'),
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _emailCtrl,
                decoration: const InputDecoration(labelText: '邮箱'),
                keyboardType: TextInputType.emailAddress,
              ),
              SizedBox(height: AppTheme.spacingM),
              TextFormField(
                controller: _phoneCtrl,
                decoration: const InputDecoration(labelText: '手机号'),
                keyboardType: TextInputType.phone,
              ),
              SizedBox(height: AppTheme.spacingM),
              DropdownButtonFormField<int?>(
                initialValue: _deptId,
                decoration: const InputDecoration(labelText: '部门'),
                items: [
                  const DropdownMenuItem(
                      value: null, child: Text('请选择部门')),
                  ..._deptOptions.map(
                    (d) => DropdownMenuItem(
                      value: d.id,
                      child: Text(d.name),
                    ),
                  ),
                ],
                onChanged: (v) => _deptId = v,
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
