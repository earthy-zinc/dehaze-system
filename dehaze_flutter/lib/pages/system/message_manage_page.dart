import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/announcement_model.dart';
import '../../models/message_template_model.dart';
import '../../providers/providers.dart';
import '../../services/announcement_service.dart';
import '../../services/message_service.dart';
import '../../services/message_template_service.dart';
import '../../theme/app_theme.dart';

// ---------------------------------------------------------------------------
// 消息列表
// ---------------------------------------------------------------------------

final messageListManageProvider = StateNotifierProvider<
    MessageListManageNotifier, AsyncValue<PagedList<Map<String, dynamic>>>>(
  (ref) => MessageListManageNotifier(ref.watch(messageServiceProvider)),
);

class MessageListManageNotifier
    extends PagedListNotifier<Map<String, dynamic>> {
  MessageListManageNotifier(this._service) : super(pageSize: 20);
  final MessageService _service;

  @override
  Future<PageResult<Map<String, dynamic>>> fetchPage(int pageNum) async {
    final response = await _service.getPage(pageNum: pageNum, pageSize: 20);
    final data = response['data'] as Map<String, dynamic>;
    final list = (data['list'] as List<dynamic>?)
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

final announcementManageProvider = StateNotifierProvider<
    AnnouncementManageNotifier, AsyncValue<PagedList<AnnouncementVO>>>(
  (ref) => AnnouncementManageNotifier(ref.watch(announcementServiceProvider)),
);

class AnnouncementManageNotifier extends PagedListNotifier<AnnouncementVO> {
  AnnouncementManageNotifier(this._service) : super();
  final AnnouncementService _service;

  @override
  Future<PageResult<AnnouncementVO>> fetchPage(int pageNum) {
    return _service.getPage(AnnouncementQuery(pageNum: pageNum, pageSize: pageSize));
  }
}

// ---------------------------------------------------------------------------
// 消息模板
// ---------------------------------------------------------------------------

final messageTemplateManageProvider = StateNotifierProvider<
    MessageTemplateManageNotifier, AsyncValue<PagedList<MessageTemplateVO>>>(
  (ref) => MessageTemplateManageNotifier(ref.watch(messageTemplateServiceProvider)),
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

/// 消息管理页面（L2）
///
/// 权限：sys:notify:*
class MessageManagePage extends ConsumerStatefulWidget {
  const MessageManagePage({super.key});

  @override
  ConsumerState<MessageManagePage> createState() => _MessageManagePageState();
}

class _MessageManagePageState extends ConsumerState<MessageManagePage>
    with SingleTickerProviderStateMixin {
  late TabController _tabCtrl;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('消息管理'),
        bottom: TabBar(controller: _tabCtrl, tabs: const [
          Tab(text: '消息列表'),
          Tab(text: '公告管理'),
          Tab(text: '消息模板'),
        ]),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: const [
          _MessageListTab(),
          _AnnouncementTab(),
          _MessageTemplateTab(),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Tab: 消息列表
// ---------------------------------------------------------------------------

class _MessageListTab extends ConsumerWidget {
  const _MessageListTab();

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
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
              controller: titleCtrl,
              decoration: const InputDecoration(labelText: '标题')),
          SizedBox(height: AppTheme.spacingM),
          TextField(
              controller: contentCtrl,
              decoration: const InputDecoration(labelText: '内容'),
              maxLines: 4),
        ]),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('取消')),
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
              child: const Text('发送')),
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
                  label: const Text('群发消息'))),
        ),
        Expanded(child: _buildBody(ref, state)),
      ],
    );
  }

  Widget _buildBody(WidgetRef ref, AsyncValue<PagedList<Map<String, dynamic>>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(messageListManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无消息'))
          : RefreshIndicator(
              onRefresh: () => ref.read(messageListManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () => ref.read(messageListManageProvider.notifier).loadMore(),
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

// ---------------------------------------------------------------------------
// Tab: 公告管理
// ---------------------------------------------------------------------------

class _AnnouncementTab extends ConsumerWidget {
  const _AnnouncementTab();

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
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
              controller: titleCtrl,
              decoration: const InputDecoration(labelText: '标题')),
          SizedBox(height: AppTheme.spacingM),
          TextField(
              controller: contentCtrl,
              decoration: const InputDecoration(labelText: '内容'),
              maxLines: 4),
        ]),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('取消')),
          FilledButton(
              onPressed: () async {
                try {
                  await ref.read(announcementServiceProvider).add(
                        AnnouncementForm(
                          title: titleCtrl.text.trim(),
                          content: contentCtrl.text.trim(),
                          type: 'NOTICE',
                          priority: 0,
                          targetType: 'ALL',
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
              child: const Text('创建')),
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
                  label: const Text('新建公告'))),
        ),
        Expanded(child: _buildBody(ref, state)),
      ],
    );
  }

  Widget _buildBody(WidgetRef ref, AsyncValue<PagedList<AnnouncementVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(announcementManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无公告'))
          : RefreshIndicator(
              onRefresh: () => ref.read(announcementManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () => ref.read(announcementManageProvider.notifier).loadMore(),
                child: ListView.builder(
                  itemCount: page.items.length,
                  itemBuilder: (context, index) {
                    final item = page.items[index];
                    return Card(
                      child: ListTile(
                        title: Text(item.title),
                        subtitle: Text(item.statusName ?? ''),
                        trailing: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconButton(
                                icon:
                                    const Icon(Icons.send, size: 20),
                                tooltip: '发送',
                                onPressed: () => _send(context, ref, item.id)),
                            IconButton(
                                icon: Icon(Icons.delete,
                                    size: 20,
                                    color: AppTheme.errorColor),
                                onPressed: () => _delete(context, ref, item.id)),
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

// ---------------------------------------------------------------------------
// Tab: 消息模板
// ---------------------------------------------------------------------------

class _MessageTemplateTab extends ConsumerWidget {
  const _MessageTemplateTab();

  void _showSnack(BuildContext context, String msg) {
    if (!context.mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _edit(BuildContext context, WidgetRef ref, MessageTemplateVO current) {
    final contentCtrl =
        TextEditingController(text: current.content);
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('编辑模板'),
        content: TextField(
            controller: contentCtrl,
            decoration: const InputDecoration(labelText: '模板内容'),
            maxLines: 4),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('取消')),
          FilledButton(
              onPressed: () async {
                try {
                  await ref.read(messageTemplateServiceProvider).update(
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
              child: const Text('保存')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(messageTemplateManageProvider);
    return _buildBody(context, ref, state);
  }

  Widget _buildBody(BuildContext context, WidgetRef ref,
      AsyncValue<PagedList<MessageTemplateVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(messageTemplateManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) => page.items.isEmpty
          ? const Center(child: Text('暂无模板'))
          : RefreshIndicator(
              onRefresh: () => ref.read(messageTemplateManageProvider.notifier).refresh(),
              child: LoadMoreListener(
                onLoadMore: () => ref.read(messageTemplateManageProvider.notifier).loadMore(),
                child: ListView.builder(
                  itemCount: page.items.length,
                  itemBuilder: (context, index) {
                    final item = page.items[index];
                    return Card(
                      child: ListTile(
                        title: Text(item.name),
                        subtitle: Text(item.content,
                            maxLines: 2, overflow: TextOverflow.ellipsis),
                        trailing: IconButton(
                            icon: const Icon(Icons.edit, size: 20),
                            onPressed: () => _edit(context, ref, item)),
                      ),
                    );
                  },
                ),
              ),
            ),
    );
  }
}
