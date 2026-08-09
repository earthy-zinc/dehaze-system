import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/feedback_model.dart';
import '../../providers/providers.dart';
import '../../services/feedback_service.dart';
import '../../theme/app_theme.dart';

final feedbackManageProvider = StateNotifierProvider<
    FeedbackManageNotifier, AsyncValue<PagedList<FeedbackPageVO>>>(
  (ref) => FeedbackManageNotifier(ref.watch(feedbackServiceProvider)),
);

class FeedbackManageNotifier extends PagedListNotifier<FeedbackPageVO> {
  FeedbackManageNotifier(this._service) : super();
  final FeedbackService _service;
  String? statusFilter;

  @override
  Future<PageResult<FeedbackPageVO>> fetchPage(int pageNum) {
    return _service.getFeedbackPage(
      FeedbackQuery(pageNum: pageNum, pageSize: pageSize, status: statusFilter),
    );
  }

  Future<void> filterByStatus(String? status) async {
    statusFilter = status;
    await refresh();
  }
}

/// 反馈评价管理页面（L2）
///
/// 权限：sys:feedback:*
class FeedbackManagePage extends ConsumerWidget {
  const FeedbackManagePage({super.key});

  void _showReply(BuildContext context, WidgetRef ref, int feedbackId) {
    final ctrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('回复反馈'),
        content: TextField(
            controller: ctrl,
            decoration: const InputDecoration(labelText: '回复内容'),
            maxLines: 3),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('取消')),
          FilledButton(
              onPressed: () async {
                try {
                  await ref.read(feedbackServiceProvider).replyFeedback(
                        FeedbackReplyForm(
                            feedbackId: feedbackId,
                            content: ctrl.text.trim()),
                      );
                  if (!c.mounted) {
                    return;
                  }
                  Navigator.pop(c);
                  _showSnack(context, '回复成功');
                  ref.read(feedbackManageProvider.notifier).refresh();
                } catch (e) {
                  _showSnack(context, extractErrorMessage(e));
                }
              },
              child: const Text('发送')),
        ],
      ),
    );
  }

  void _closeFeedback(BuildContext context, WidgetRef ref, int feedbackId) {
    final ctrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('关闭反馈'),
        content: TextField(
            controller: ctrl,
            decoration: const InputDecoration(labelText: '关闭原因')),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('取消')),
          FilledButton(
              onPressed: () async {
                try {
                  await ref.read(feedbackServiceProvider).closeFeedback(
                        FeedbackCloseForm(
                            feedbackId: feedbackId,
                            reason: ctrl.text.trim()),
                      );
                  if (!c.mounted) {
                    return;
                  }
                  Navigator.pop(c);
                  _showSnack(context, '已关闭');
                  ref.read(feedbackManageProvider.notifier).refresh();
                } catch (e) {
                  _showSnack(context, extractErrorMessage(e));
                }
              },
              child: const Text('确定')),
        ],
      ),
    );
  }

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final state = ref.watch(feedbackManageProvider);
    final statusFilter = ref.watch(feedbackManageProvider.notifier).statusFilter;

    return Scaffold(
      appBar: AppBar(title: const Text('反馈评价管理')),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.symmetric(
                horizontal: AppTheme.spacingM, vertical: AppTheme.spacingS),
            child: Row(children: [
              Expanded(
                child: DropdownButtonFormField<String?>(
                  initialValue: statusFilter,
                  decoration: const InputDecoration(
                      labelText: '状态筛选', isDense: true),
                  items: const [
                    DropdownMenuItem(value: null, child: Text('全部')),
                    DropdownMenuItem(value: 'OPEN', child: Text('待处理')),
                    DropdownMenuItem(
                        value: 'IN_PROGRESS', child: Text('处理中')),
                    DropdownMenuItem(value: 'CLOSED', child: Text('已关闭')),
                  ],
                  onChanged: (v) =>
                      ref.read(feedbackManageProvider.notifier).filterByStatus(v),
                ),
              ),
            ]),
          ),
          Expanded(child: _buildBody(theme, ref, state)),
        ],
      ),
    );
  }

  Widget _buildBody(ThemeData theme, WidgetRef ref,
      AsyncValue<PagedList<FeedbackPageVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
        Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
        SizedBox(height: AppTheme.spacingM),
        FilledButton(
            onPressed: () => ref.read(feedbackManageProvider.notifier).refresh(),
            child: const Text('重试')),
      ])),
      data: (page) {
        if (page.items.isEmpty) {
          return const Center(child: Text('暂无数据'));
        }
        return RefreshIndicator(
          onRefresh: () => ref.read(feedbackManageProvider.notifier).refresh(),
          child: LoadMoreListener(
            onLoadMore: () => ref.read(feedbackManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                return Card(
                  child: ListTile(
                    title: Text(item.title),
                    subtitle: Text(
                        '${item.statusName ?? item.status} | ${item.createTime}'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                            icon: const Icon(Icons.reply, size: 20),
                            tooltip: '回复',
                            onPressed: () => _showReply(context, ref, item.id)),
                        IconButton(
                            icon: Icon(Icons.archive,
                                size: 20, color: AppTheme.gray500),
                            tooltip: '关闭',
                            onPressed: () =>
                                _closeFeedback(context, ref, item.id)),
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
