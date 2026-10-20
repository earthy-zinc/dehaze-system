import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/message_template_model.dart';
import '../../providers/providers.dart';
import '../../services/message_template_service.dart';
import '../../theme/app_theme.dart';

final messageTemplateManageProvider =
    StateNotifierProvider<
      MessageTemplateManageNotifier,
      AsyncValue<PagedList<MessageTemplateVO>>
    >(
      (ref) => MessageTemplateManageNotifier(
        ref.watch(messageTemplateServiceProvider),
      ),
    );

class MessageTemplateManageNotifier
    extends PagedListNotifier<MessageTemplateVO> {
  MessageTemplateManageNotifier(this._service) : super();
  final MessageTemplateService _service;

  @override
  Future<PageResult<MessageTemplateVO>> fetchPage(int pageNum) {
    return _service.getPage(
      MessageTemplateQuery(pageNum: pageNum, pageSize: pageSize),
    );
  }
}

// ---------------------------------------------------------------------------
// 主页面
// ---------------------------------------------------------------------------

class MessageTemplateTab extends ConsumerWidget {
  const MessageTemplateTab({super.key});

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _edit(BuildContext context, WidgetRef ref, MessageTemplateVO current) {
    final contentCtrl = TextEditingController(text: current.content);
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('编辑模板'),
        content: TextField(
          controller: contentCtrl,
          decoration: const InputDecoration(labelText: '模板内容'),
          maxLines: 4,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(c),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () async {
              try {
                await ref
                    .read(messageTemplateServiceProvider)
                    .update(
                      current.id,
                      MessageTemplateForm(
                        code: current.code,
                        name: current.name,
                        title: current.title,
                        content: contentCtrl.text.trim(),
                        type: current.type,
                        status: current.status,
                        variables: current.variables,
                        description: current.description,
                      ),
                    );
                if (!c.mounted) {
                  return;
                }
                Navigator.pop(c);
                _showSnack(context, '更新成功');
                ref.read(messageTemplateManageProvider.notifier).refresh();
              } catch (e) {
                _showSnack(context, extractErrorMessage(e));
              }
            },
            child: const Text('保存'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(messageTemplateManageProvider);
    return _buildBody(context, ref, state);
  }

  Widget _buildBody(
    BuildContext context,
    WidgetRef ref,
    AsyncValue<PagedList<MessageTemplateVO>> state,
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
                  ref.read(messageTemplateManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无模板'))
          : RefreshIndicator(
              onRefresh: () =>
                  ref.read(messageTemplateManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () =>
                    ref.read(messageTemplateManageProvider.notifier).loadMore(),
                child: ListView.builder(
                  itemCount: page.items.length,
                  itemBuilder: (context, index) {
                    final item = page.items[index];
                    return Card(
                      child: ListTile(
                        title: Text(item.name),
                        subtitle: Text(
                          item.content,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        trailing: IconButton(
                          icon: const Icon(Icons.edit, size: 20),
                          onPressed: () => _edit(context, ref, item),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
    );
  }
}
