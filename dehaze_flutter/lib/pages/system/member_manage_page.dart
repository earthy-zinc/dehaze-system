import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/member_model.dart';
import '../../providers/providers.dart';
import '../../services/member_service.dart';
import '../../theme/app_theme.dart';

final memberManageProvider =
    StateNotifierProvider<MemberManageNotifier, AsyncValue<PagedList<MemberPageVO>>>(
  (ref) => MemberManageNotifier(ref.watch(memberServiceProvider)),
);

class MemberManageNotifier extends PagedListNotifier<MemberPageVO> {
  MemberManageNotifier(this._service) : super();
  final MemberService _service;

  @override
  Future<PageResult<MemberPageVO>> fetchPage(int pageNum) {
    return _service.getPage(
      MemberQuery(pageNum: pageNum, pageSize: 10, keywords: keyword),
    );
  }
}

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

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  MemberService get _service => ref.read(memberServiceProvider);

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
                ref.read(memberManageProvider.notifier).refresh();
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
                ref.read(memberManageProvider.notifier).refresh();
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
    final theme = Theme.of(context);
    final state = ref.watch(memberManageProvider);

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
                    ref.read(memberManageProvider.notifier).search('');
                  },
                ),
              ),
              onSubmitted: (v) => ref.read(memberManageProvider.notifier).search(v),
            ),
          ),
          Expanded(
            child: state.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
                    SizedBox(height: AppTheme.spacingM),
                    FilledButton(
                      onPressed: () => ref.read(memberManageProvider.notifier).refresh(),
                      child: const Text('重试'),
                    ),
                  ],
                ),
              ),
              data: (page) => page.items.isEmpty
                  ? const Center(child: Text('暂无数据'))
                  : RefreshIndicator(
                      onRefresh: () => ref.read(memberManageProvider.notifier).refresh(),
                      child: LoadMoreListener(
                        onLoadMore: () => ref.read(memberManageProvider.notifier).loadMore(),
                        child: ListView.builder(
                          itemCount: page.items.length,
                          itemBuilder: (context, index) {
                            final item = page.items[index];
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
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
