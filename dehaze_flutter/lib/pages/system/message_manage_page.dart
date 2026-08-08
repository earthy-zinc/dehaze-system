import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/announcement_model.dart';
import '../../models/message_template_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

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
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:notify:*')) {
      return Scaffold(
          appBar: AppBar(title: const Text('消息管理')),
          body: const Center(child: Text('无权限访问')));
    }

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

class _MessageListTab extends ConsumerStatefulWidget {
  const _MessageListTab();

  @override
  ConsumerState<_MessageListTab> createState() => _MessageListTabState();
}

class _MessageListTabState extends ConsumerState<_MessageListTab> {
  List<Map<String, dynamic>> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;

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
    });
    try {
      final result = await ref
          .read(messageServiceProvider)
          .getPage(pageNum: _pageNum, pageSize: 20);
      final data = result['data'] as Map<String, dynamic>;
      final list = (data['list'] as List<dynamic>?)
              ?.map((e) => e as Map<String, dynamic>)
              .toList() ??
          [];
      setState(() {
        if (reset) {
          _items = list;
        } else {
          _items.addAll(list);
        }
        _total = (data['total'] as int?) ?? 0;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
      });
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showSendDialog() {
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
                  _showSnack('发送成功');
                  _fetchData(reset: true);
                } catch (e) {
                  _showSnack(extractErrorMessage(e));
                }
              },
              child: const Text('发送')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                  onPressed: _showSendDialog,
                  icon: const Icon(Icons.send),
                  label: const Text('群发消息'))),
        ),
        Expanded(
            child: _loading && _items.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : _items.isEmpty
                    ? const Center(child: Text('暂无消息'))
                    : RefreshIndicator(
                        onRefresh: () => _fetchData(reset: true),
                        child: ListView.builder(
                          itemCount: _items.length +
                              (_items.length < _total ? 1 : 0),
                          itemBuilder: (context, index) {
                            if (index >= _items.length) {
                              if (!_loading) {
                                _pageNum++;
                                _fetchData();
                              }
                              return const Center(
                                  child: Padding(
                                      padding: EdgeInsets.all(16),
                                      child: CircularProgressIndicator()));
                            }
                            final item = _items[index];
                            return Card(
                              child: ListTile(
                                title: Text(item['title'] as String? ?? ''),
                                subtitle: Text(item['type'] as String? ?? ''),
                              ),
                            );
                          },
                        ),
                      )),
      ],
    );
  }
}

class _AnnouncementTab extends ConsumerStatefulWidget {
  const _AnnouncementTab();

  @override
  ConsumerState<_AnnouncementTab> createState() => _AnnouncementTabState();
}

class _AnnouncementTabState extends ConsumerState<_AnnouncementTab> {
  List<AnnouncementVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
  }

  Future<void> _fetchData({bool reset = false}) async {
    if (reset) {
      _pageNum = 1;
    }
    setState(() => _loading = true);
    try {
      final result = await ref
          .read(announcementServiceProvider)
          .getPage(AnnouncementQuery(pageNum: _pageNum, pageSize: 10));
      setState(() {
        if (reset) {
          _items = result.list;
        } else {
          _items.addAll(result.list);
        }
        _total = result.total;
        _loading = false;
      });
    } catch (_) {
      setState(() => _loading = false);
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showForm() {
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
                  _showSnack('创建成功');
                  _fetchData(reset: true);
                } catch (e) {
                  _showSnack(extractErrorMessage(e));
                }
              },
              child: const Text('创建')),
        ],
      ),
    );
  }

  Future<void> _send(int id) async {
    try {
      await ref.read(announcementServiceProvider).send(id);
      _showSnack('发送成功');
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  Future<void> _delete(int id) async {
    try {
      await ref.read(announcementServiceProvider).delete(id);
      _showSnack('删除成功');
      _fetchData(reset: true);
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: EdgeInsets.all(AppTheme.spacingM),
          child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                  onPressed: _showForm,
                  icon: const Icon(Icons.add),
                  label: const Text('新建公告'))),
        ),
        Expanded(
            child: _loading && _items.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : _items.isEmpty
                    ? const Center(child: Text('暂无公告'))
                    : RefreshIndicator(
                        onRefresh: () => _fetchData(reset: true),
                        child: ListView.builder(
                          itemCount: _items.length +
                              (_items.length < _total ? 1 : 0),
                          itemBuilder: (context, index) {
                            if (index >= _items.length) {
                              if (!_loading) {
                                _pageNum++;
                                _fetchData();
                              }
                              return const Center(
                                  child: Padding(
                                      padding: EdgeInsets.all(16),
                                      child: CircularProgressIndicator()));
                            }
                            final item = _items[index];
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
                                        onPressed: () => _send(item.id)),
                                    IconButton(
                                        icon: Icon(Icons.delete,
                                            size: 20,
                                            color: AppTheme.errorColor),
                                        onPressed: () => _delete(item.id)),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                      )),
      ],
    );
  }
}

class _MessageTemplateTab extends ConsumerStatefulWidget {
  const _MessageTemplateTab();

  @override
  ConsumerState<_MessageTemplateTab> createState() =>
      _MessageTemplateTabState();
}

class _MessageTemplateTabState extends ConsumerState<_MessageTemplateTab> {
  List<MessageTemplateVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
  }

  Future<void> _fetchData({bool reset = false}) async {
    if (reset) {
      _pageNum = 1;
    }
    setState(() => _loading = true);
    try {
      final result = await ref
          .read(messageTemplateServiceProvider)
          .getPage(MessageTemplateQuery(pageNum: _pageNum, pageSize: 10));
      setState(() {
        if (reset) {
          _items = result.list;
        } else {
          _items.addAll(result.list);
        }
        _total = result.total;
        _loading = false;
      });
    } catch (_) {
      setState(() => _loading = false);
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _edit(int id) {
    final current =
        _items.firstWhere((e) => e.id == id);
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
                        id,
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
                  _showSnack('更新成功');
                  _fetchData(reset: true);
                } catch (e) {
                  _showSnack(extractErrorMessage(e));
                }
              },
              child: const Text('保存')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) => _loading && _items.isEmpty
      ? const Center(child: CircularProgressIndicator())
      : _items.isEmpty
          ? const Center(child: Text('暂无模板'))
          : RefreshIndicator(
              onRefresh: () => _fetchData(reset: true),
              child: ListView.builder(
                itemCount: _items.length + (_items.length < _total ? 1 : 0),
                itemBuilder: (context, index) {
                  if (index >= _items.length) {
                    if (!_loading) {
                      _pageNum++;
                      _fetchData();
                    }
                    return const Center(
                        child: Padding(
                            padding: EdgeInsets.all(16),
                            child: CircularProgressIndicator()));
                  }
                  final item = _items[index];
                  return Card(
                    child: ListTile(
                      title: Text(item.name),
                      subtitle: Text(item.content,
                          maxLines: 2, overflow: TextOverflow.ellipsis),
                      trailing: IconButton(
                          icon: const Icon(Icons.edit, size: 20),
                          onPressed: () => _edit(item.id)),
                    ),
                  );
                },
              ),
            );
}
