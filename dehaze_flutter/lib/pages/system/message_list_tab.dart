import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../providers/providers.dart';
import '../../services/message_service.dart';
import '../../theme/app_theme.dart';

final messageListManageProvider =
    StateNotifierProvider<
      MessageListManageNotifier,
      AsyncValue<PagedList<Map<String, dynamic>>>
    >((ref) => MessageListManageNotifier(ref.watch(messageServiceProvider)));

class MessageListManageNotifier
    extends PagedListNotifier<Map<String, dynamic>> {
  MessageListManageNotifier(this._service) : super(pageSize: 20);
  final MessageService _service;

  @override
  Future<PageResult<Map<String, dynamic>>> fetchPage(int pageNum) async {
    final response = await _service.getPage(pageNum: pageNum, pageSize: 20);
    final data = response['data'] as Map<String, dynamic>;
    final list =
        (data['list'] as List<dynamic>?)
            ?.map((e) => e as Map<String, dynamic>)
            .toList() ??
        [];
    return PageResult<Map<String, dynamic>>(
      list: list,
      total: (data['total'] as num?)?.toInt() ?? 0,
    );
  }
}

// ---------------------------------------------------------------------------
// 公告管理
// ---------------------------------------------------------------------------

class MessageListTab extends ConsumerWidget {
  const MessageListTab({super.key});

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showSendDialog(BuildContext context, WidgetRef ref) {
    final titleCtrl = TextEditingController();
    final contentCtrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('群发消息'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: titleCtrl,
              decoration: const InputDecoration(labelText: '标题'),
            ),
            SizedBox(height: AppTheme.spacingM),
            TextField(
              controller: contentCtrl,
              decoration: const InputDecoration(labelText: '内容'),
              maxLines: 4,
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(c),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () async {
              try {
                await ref.read(messageServiceProvider).send({
                  'title': titleCtrl.text.trim(),
                  'content': contentCtrl.text.trim(),
                });
                if (!c.mounted) {
                  return;
                }
                Navigator.pop(c);
                _showSnack(context, '发送成功');
                ref.read(messageListManageProvider.notifier).refresh();
              } catch (e) {
                _showSnack(context, extractErrorMessage(e));
              }
            },
            child: const Text('发送'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(messageListManageProvider);
    return Column(
      children: [
        Padding(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => _showSendDialog(context, ref),
              icon: const Icon(Icons.send),
              label: const Text('群发消息'),
            ),
          ),
        ),
        Expanded(child: _buildBody(ref, state)),
      ],
    );
  }

  Widget _buildBody(
    WidgetRef ref,
    AsyncValue<PagedList<Map<String, dynamic>>> state,
  ) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () =>
                  ref.read(messageListManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无消息'))
          : RefreshIndicator(
              onRefresh: () =>
                  ref.read(messageListManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () =>
                    ref.read(messageListManageProvider.notifier).loadMore(),
                child: ListView.builder(
                  itemCount: page.items.length,
                  itemBuilder: (context, index) {
                    final item = page.items[index];
                    return Card(
                      child: ListTile(
                        title: Text(item['title'] as String? ?? ''),
                        subtitle: Text(item['type'] as String? ?? ''),
                      ),
                    );
                  },
                ),
              ),
            ),
    );
  }
}
