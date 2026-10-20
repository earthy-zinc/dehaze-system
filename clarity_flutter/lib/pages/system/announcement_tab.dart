import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/announcement_model.dart';
import '../../providers/providers.dart';
import '../../services/announcement_service.dart';
import '../../theme/app_theme.dart';

final announcementManageProvider =
    StateNotifierProvider<
      AnnouncementManageNotifier,
      AsyncValue<PagedList<AnnouncementVO>>
    >(
      (ref) =>
          AnnouncementManageNotifier(ref.watch(announcementServiceProvider)),
    );

class AnnouncementManageNotifier extends PagedListNotifier<AnnouncementVO> {
  AnnouncementManageNotifier(this._service) : super();
  final AnnouncementService _service;

  @override
  Future<PageResult<AnnouncementVO>> fetchPage(int pageNum) {
    return _service.getPage(
      AnnouncementQuery(pageNum: pageNum, pageSize: pageSize),
    );
  }
}

// ---------------------------------------------------------------------------
// 消息模板
// ---------------------------------------------------------------------------

class AnnouncementTab extends ConsumerWidget {
  const AnnouncementTab({super.key});

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showForm(BuildContext context, WidgetRef ref) {
    final titleCtrl = TextEditingController();
    final contentCtrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('新建公告'),
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
                await ref
                    .read(announcementServiceProvider)
                    .add(
                      AnnouncementForm(
                        title: titleCtrl.text.trim(),
                        content: contentCtrl.text.trim(),
                        type: 'NOTICE',
                        importance: 1,
                        targetScope: AnnouncementTargetScope.all,
                      ),
                    );
                if (!c.mounted) {
                  return;
                }
                Navigator.pop(c);
                _showSnack(context, '创建成功');
                ref.read(announcementManageProvider.notifier).refresh();
              } catch (e) {
                _showSnack(context, extractErrorMessage(e));
              }
            },
            child: const Text('创建'),
          ),
        ],
      ),
    );
  }

  Future<void> _send(BuildContext context, WidgetRef ref, int id) async {
    try {
      await ref.read(announcementServiceProvider).send(id);
      if (!context.mounted) return;
      _showSnack(context, '发送成功');
    } catch (e) {
      if (!context.mounted) return;
      _showSnack(context, extractErrorMessage(e));
    }
  }

  Future<void> _delete(BuildContext context, WidgetRef ref, int id) async {
    try {
      await ref.read(announcementServiceProvider).delete(id);
      ref.read(announcementManageProvider.notifier).refresh();
      if (!context.mounted) return;
      _showSnack(context, '删除成功');
    } catch (e) {
      if (!context.mounted) return;
      _showSnack(context, extractErrorMessage(e));
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(announcementManageProvider);
    return Column(
      children: [
        Padding(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => _showForm(context, ref),
              icon: const Icon(Icons.add),
              label: const Text('新建公告'),
            ),
          ),
        ),
        Expanded(child: _buildBody(ref, state)),
      ],
    );
  }

  Widget _buildBody(
    WidgetRef ref,
    AsyncValue<PagedList<AnnouncementVO>> state,
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
                  ref.read(announcementManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无公告'))
          : RefreshIndicator(
              onRefresh: () =>
                  ref.read(announcementManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () =>
                    ref.read(announcementManageProvider.notifier).loadMore(),
                child: ListView.builder(
                  itemCount: page.items.length,
                  itemBuilder: (context, index) {
                    final item = page.items[index];
                    return Card(
                      child: ListTile(
                        title: Text(item.title),
                        subtitle: Text(item.statusLabel ?? ''),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                              icon: const Icon(Icons.send, size: 20),
                              tooltip: '发送',
                              onPressed: () => _send(context, ref, item.id),
                            ),
                            IconButton(
                              icon: Icon(
                                Icons.delete,
                                size: 20,
                                color: AppTheme.errorColor,
                              ),
                              onPressed: () => _delete(context, ref, item.id),
                            ),
                          ],
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
