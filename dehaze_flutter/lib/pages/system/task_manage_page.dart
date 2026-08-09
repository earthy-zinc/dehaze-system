import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/task_model.dart';
import '../../providers/providers.dart';
import '../../services/task_service.dart';
import '../../theme/app_theme.dart';

final taskManageProvider =
    StateNotifierProvider<TaskManageNotifier, AsyncValue<PagedList<TaskVO>>>(
  (ref) => TaskManageNotifier(ref.watch(taskServiceProvider)),
);

class TaskManageNotifier extends PagedListNotifier<TaskVO> {
  TaskManageNotifier(this._service) : super(pageSize: 20);
  final TaskService _service;

  TaskStatusType? _status;
  String? _taskType;
  TaskCategory? _taskCategory;

  @override
  Future<PageResult<TaskVO>> fetchPage(int pageNum) {
    return _service.getPage(TaskQuery(
      pageNum: pageNum,
      pageSize: 20,
      status: _status,
      taskType: _taskType,
      taskCategory: _taskCategory,
    ));
  }

  Future<void> applyFilters({
    TaskStatusType? status,
    String? taskType,
    TaskCategory? taskCategory,
  }) async {
    _status = status;
    _taskType = taskType;
    _taskCategory = taskCategory;
    await refresh();
  }
}

const _statusLabels = {
  TaskStatusType.pending: '待执行',
  TaskStatusType.processing: '执行中',
  TaskStatusType.completed: '已完成',
  TaskStatusType.failed: '失败',
  TaskStatusType.cancelled: '已取消',
};

/// 任务管理页面（L2）
///
/// 权限：sys:task:*
class TaskManagePage extends ConsumerWidget {
  const TaskManagePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final state = ref.watch(taskManageProvider);

    Future<void> cancel(String taskId) async {
      try {
        await ref.read(taskServiceProvider).cancel(taskId);
        ref.read(taskManageProvider.notifier).refresh();
        if (!context.mounted) return;
        _showSnack(context, '已取消');
      } catch (e) {
        if (!context.mounted) return;
        _showSnack(context, extractErrorMessage(e));
      }
    }

    Future<void> retry(String taskId) async {
      try {
        await ref.read(taskServiceProvider).retry(taskId);
        ref.read(taskManageProvider.notifier).refresh();
        if (!context.mounted) return;
        _showSnack(context, '已重试');
      } catch (e) {
        if (!context.mounted) return;
        _showSnack(context, extractErrorMessage(e));
      }
    }

    return Scaffold(
      appBar: AppBar(title: const Text('任务管理')),
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
                onPressed: () => ref.read(taskManageProvider.notifier).refresh(),
                child: const Text('重试'),
              ),
            ],
          ),
        ),
        data: (page) => page.items.isEmpty
            ? const Center(child: Text('暂无数据'))
            : RefreshIndicator(
                onRefresh: () => ref.read(taskManageProvider.notifier).refresh(),
                child: LoadMoreListener(
                  onLoadMore: () => ref.read(taskManageProvider.notifier).loadMore(),
                  child: ListView.builder(
                    itemCount: page.items.length,
                    itemBuilder: (context, index) {
                      final item = page.items[index];
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
                                  onPressed: () => cancel(item.taskId),
                                ),
                              if (item.status == TaskStatusType.failed)
                                IconButton(
                                  icon: Icon(
                                    Icons.refresh,
                                    size: 20,
                                    color: AppTheme.infoColor,
                                  ),
                                  tooltip: '重试',
                                  onPressed: () => retry(item.taskId),
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

void _showSnack(BuildContext context, String msg) {
  if (!context.mounted) {
    return;
  }
  ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
}
