import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/task_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 任务管理页面（L2）
///
/// 权限：sys:task:*
class TaskManagePage extends ConsumerStatefulWidget {
  const TaskManagePage({super.key});

  @override
  ConsumerState<TaskManagePage> createState() => _TaskManagePageState();
}

class _TaskManagePageState extends ConsumerState<TaskManagePage> {
  List<TaskVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
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
      final result = await ref.read(taskServiceProvider).getPage(
        TaskQuery(pageNum: _pageNum, pageSize: 20),
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

  Future<void> _cancel(String taskId) async {
    try {
      await ref.read(taskServiceProvider).cancel(taskId);
      _showSnack('已取消');
      _fetchData(reset: true);
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _retry(String taskId) async {
    try {
      await ref.read(taskServiceProvider).retry(taskId);
      _showSnack('已重试');
      _fetchData(reset: true);
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

  static const _statusLabels = {
    TaskStatusType.pending: '待执行',
    TaskStatusType.processing: '执行中',
    TaskStatusType.completed: '已完成',
    TaskStatusType.failed: '失败',
    TaskStatusType.cancelled: '已取消',
  };

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:task:*')) {
      return Scaffold(
        appBar: AppBar(title: const Text('任务管理')),
        body: const Center(child: Text('无权限访问')),
      );
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('任务管理')),
      body:
          _loading && _items.isEmpty
              ? const Center(child: CircularProgressIndicator())
              : _error != null
              ? Center(
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
              )
              : _items.isEmpty
              ? const Center(child: Text('暂无数据'))
              : RefreshIndicator(
                onRefresh: () => _fetchData(reset: true),
                child: ListView.builder(
                  itemCount:
                      _items.length + (_items.length < _total ? 1 : 0),
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
                    final label =
                        _statusLabels[item.status] ?? item.status.name;
                    final isActive =
                        item.status == TaskStatusType.pending ||
                        item.status == TaskStatusType.processing;
                    return Card(
                      child: ListTile(
                        title: Text(item.taskType ?? ''),
                        subtitle: Text(
                          'ID: ${item.taskId} | 状态: $label',
                        ),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            if (isActive)
                              IconButton(
                                icon: Icon(
                                  Icons.cancel,
                                  size: 20,
                                  color: AppTheme.errorColor,
                                ),
                                tooltip: '取消',
                                onPressed: () => _cancel(item.taskId),
                              ),
                            if (item.status == TaskStatusType.failed)
                              IconButton(
                                icon: Icon(
                                  Icons.refresh,
                                  size: 20,
                                  color: AppTheme.infoColor,
                                ),
                                tooltip: '重试',
                                onPressed: () => _retry(item.taskId),
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
    );
  }
}
