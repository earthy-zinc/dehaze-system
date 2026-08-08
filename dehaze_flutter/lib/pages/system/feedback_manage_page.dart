import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/feedback_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 反馈评价管理页面（L2）
///
/// 权限：sys:feedback:*
class FeedbackManagePage extends ConsumerStatefulWidget {
  const FeedbackManagePage({super.key});

  @override
  ConsumerState<FeedbackManagePage> createState() => _FeedbackManagePageState();
}

class _FeedbackManagePageState extends ConsumerState<FeedbackManagePage> {
  List<FeedbackPageVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;
  String? _error;
  String? _statusFilter;

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
      final result = await ref.read(feedbackServiceProvider).getFeedbackPage(
            FeedbackQuery(pageNum: _pageNum, pageSize: 10, status: _statusFilter),
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

  void _showReply(int feedbackId) {
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
                  _showSnack('回复成功');
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

  void _closeFeedback(int feedbackId) {
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
                  _showSnack('已关闭');
                  _fetchData(reset: true);
                } catch (e) {
                  _showSnack(extractErrorMessage(e));
                }
              },
              child: const Text('确定')),
        ],
      ),
    );
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:feedback:*')) {
      return Scaffold(
          appBar: AppBar(title: const Text('反馈评价管理')),
          body: const Center(child: Text('无权限访问')));
    }
    final theme = Theme.of(context);

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
                  initialValue: _statusFilter,
                  decoration: const InputDecoration(
                      labelText: '状态筛选', isDense: true),
                  items: const [
                    DropdownMenuItem(value: null, child: Text('全部')),
                    DropdownMenuItem(value: 'OPEN', child: Text('待处理')),
                    DropdownMenuItem(
                        value: 'IN_PROGRESS', child: Text('处理中')),
                    DropdownMenuItem(value: 'CLOSED', child: Text('已关闭')),
                  ],
                  onChanged: (v) {
                    _statusFilter = v;
                    _fetchData(reset: true);
                  },
                ),
              ),
            ]),
          ),
          Expanded(child: _buildList(theme)),
        ],
      ),
    );
  }

  Widget _buildList(ThemeData theme) {
    if (_loading && _items.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
        Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
        SizedBox(height: AppTheme.spacingM),
        FilledButton(
            onPressed: () => _fetchData(reset: true),
            child: const Text('重试')),
      ]));
    }
    if (_items.isEmpty) {
      return const Center(child: Text('暂无数据'));
    }

    return RefreshIndicator(
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
                child:
                    Padding(padding: EdgeInsets.all(16), child: CircularProgressIndicator()));
          }
          final item = _items[index];
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
                      onPressed: () => _showReply(item.id)),
                  IconButton(
                      icon: Icon(Icons.archive,
                          size: 20, color: AppTheme.gray500),
                      tooltip: '关闭',
                      onPressed: () => _closeFeedback(item.id)),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
