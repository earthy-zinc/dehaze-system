import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/member_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/member_service.dart';
import '../../theme/app_theme.dart';

/// 会员管理页面（L2）
///
/// 权限：sys:member:*
class MemberManagePage extends ConsumerStatefulWidget {
  const MemberManagePage({super.key});

  @override
  ConsumerState<MemberManagePage> createState() => _MemberManagePageState();
}

class _MemberManagePageState extends ConsumerState<MemberManagePage> {
  final _searchController = TextEditingController();
  List<MemberPageVO> _items = [];
  int _total = 0;
  int _pageNum = 1;
  bool _loading = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchData());
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  MemberService get _service => ref.read(memberServiceProvider);

  Future<void> _fetchData({bool reset = false}) async {
    if (reset) {
      _pageNum = 1;
    }
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final result = await _service.getPage(
        MemberQuery(
          pageNum: _pageNum,
          pageSize: 10,
          keywords: _searchController.text,
        ),
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

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  void _showLevelAdjust(int userId) {
    final ctrl = TextEditingController();
    final reasonCtrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('调整等级'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: ctrl,
              decoration: const InputDecoration(labelText: '新等级代码'),
              keyboardType: TextInputType.text,
            ),
            SizedBox(height: AppTheme.spacingM),
            TextField(
              controller: reasonCtrl,
              decoration: const InputDecoration(labelText: '原因'),
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
                await _service.adjustLevel(
                  userId,
                  MemberLevelAdjustForm(
                    levelCode: ctrl.text.trim(),
                    reason: reasonCtrl.text.trim(),
                  ),
                );
                if (!c.mounted) {
                  return;
                }
                Navigator.pop(c);
                _showSnack('等级已调整');
                _fetchData(reset: true);
              } catch (e) {
                if (!mounted) {
                  return;
                }
                _showSnack(extractErrorMessage(e));
              }
            },
            child: const Text('确定'),
          ),
        ],
      ),
    );
  }

  void _showGrowthAdjust(int userId) {
    final ctrl = TextEditingController();
    final reasonCtrl = TextEditingController();
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('调整成长值'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: ctrl,
              decoration: const InputDecoration(labelText: '成长值变动'),
              keyboardType: TextInputType.number,
            ),
            SizedBox(height: AppTheme.spacingM),
            TextField(
              controller: reasonCtrl,
              decoration: const InputDecoration(labelText: '原因'),
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
                await _service.adjustGrowth(
                  userId,
                  MemberGrowthAdjustForm(
                    changeValue: int.tryParse(ctrl.text) ?? 0,
                    reason: reasonCtrl.text.trim(),
                  ),
                );
                if (!c.mounted) {
                  return;
                }
                Navigator.pop(c);
                _showSnack('成长值已调整');
                _fetchData(reset: true);
              } catch (e) {
                if (!mounted) {
                  return;
                }
                _showSnack(extractErrorMessage(e));
              }
            },
            child: const Text('确定'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:member:*')) {
      return Scaffold(
        appBar: AppBar(title: const Text('会员管理')),
        body: const Center(child: Text('无权限访问')),
      );
    }
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('会员管理')),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(AppTheme.spacingM),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: '搜索用户名',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    _fetchData(reset: true);
                  },
                ),
              ),
              onSubmitted: (_) => _fetchData(reset: true),
            ),
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
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => _fetchData(reset: true),
              child: const Text('重试'),
            ),
          ],
        ),
      );
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
              child: Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              ),
            );
          }
          final item = _items[index];
          return Card(
            child: ListTile(
              leading: const Icon(Icons.card_membership),
              title: Text(item.nickname.isNotEmpty ? item.nickname : item.username),
              subtitle: Text('等级: ${item.levelName} | 成长值: ${item.growthValue}'),
              trailing: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.stars, size: 20),
                    tooltip: '调整等级',
                    onPressed: () => _showLevelAdjust(item.userId),
                  ),
                  IconButton(
                    icon: const Icon(Icons.trending_up, size: 20),
                    tooltip: '调整成长值',
                    onPressed: () => _showGrowthAdjust(item.userId),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
