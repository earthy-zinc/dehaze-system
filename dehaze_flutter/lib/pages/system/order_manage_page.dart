import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../core/network/page_result.dart';
import '../../core/state/paged_list_notifier.dart';
import '../../models/order_model.dart';
import '../../providers/providers.dart';
import '../../services/order_service.dart';
import '../../theme/app_theme.dart';

final orderManageProvider =
    StateNotifierProvider<OrderManageNotifier, AsyncValue<PagedList<OrderPageVO>>>(
  (ref) => OrderManageNotifier(ref.watch(orderServiceProvider)),
);

class OrderManageNotifier extends PagedListNotifier<OrderPageVO> {
  OrderManageNotifier(this._service) : super();
  final OrderService _service;

  @override
  Future<PageResult<OrderPageVO>> fetchPage(int pageNum) {
    return _service.getPage(
      OrderQuery(pageNum: pageNum, pageSize: pageSize, keyword: keyword),
    );
  }
}

final refundManageProvider = StateNotifierProvider<
    RefundManageNotifier, AsyncValue<PagedList<RefundRecordVO>>>(
  (ref) => RefundManageNotifier(ref.watch(orderServiceProvider)),
);

class RefundManageNotifier extends PagedListNotifier<RefundRecordVO> {
  RefundManageNotifier(this._service) : super();
  final OrderService _service;

  @override
  Future<PageResult<RefundRecordVO>> fetchPage(int pageNum) {
    return _service.getRefundPage(
      RefundQuery(pageNum: pageNum, pageSize: pageSize),
    );
  }
}

/// 订单管理页面（L2）
///
/// 权限：sys:order:*
class OrderManagePage extends ConsumerStatefulWidget {
  const OrderManagePage({super.key});

  @override
  ConsumerState<OrderManagePage> createState() => _OrderManagePageState();
}

class _OrderManagePageState extends ConsumerState<OrderManagePage>
    with SingleTickerProviderStateMixin {
  late TabController _tabCtrl;
  final _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    _searchController.dispose();
    super.dispose();
  }

  void _searchCurrent(String v) {
    if (_tabCtrl.index == 0) {
      ref.read(orderManageProvider.notifier).search(v);
    } else {
      ref.read(refundManageProvider.notifier).search(v);
    }
  }

  Future<void> _auditRefund(int refundId, bool approved) async {
    final reasonCtrl = TextEditingController();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: Text(approved ? '通过退款' : '驳回退款'),
        content: TextField(
          controller: reasonCtrl,
          decoration: const InputDecoration(labelText: '审核意见'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(c, false),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(c, true),
            child: const Text('确定'),
          ),
        ],
      ),
    );
    if (confirmed != true) {
      return;
    }
    try {
      await ref.read(orderServiceProvider).auditRefund(
        RefundAuditForm(
          refundId: refundId,
          approved: approved,
          remark: reasonCtrl.text.trim(),
        ),
      );
      _showSnack('审核完成');
      ref.read(refundManageProvider.notifier).refresh();
    } catch (e) {
      _showSnack(extractErrorMessage(e));
    }
  }

  void _showSnack(String msg) {
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final orderState = ref.watch(orderManageProvider);
    final refundState = ref.watch(refundManageProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('订单管理'),
        bottom: TabBar(
          controller: _tabCtrl,
          tabs: const [Tab(text: '订单列表'), Tab(text: '退款审核')],
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(AppTheme.spacingM),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: '搜索订单号',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    _searchCurrent('');
                  },
                ),
              ),
              onSubmitted: (v) => _searchCurrent(v),
            ),
          ),
          Expanded(
            child: TabBarView(
              controller: _tabCtrl,
              children: [
                _buildOrderList(theme, orderState),
                _buildRefundList(theme, refundState),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOrderList(ThemeData theme, AsyncValue<PagedList<OrderPageVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(orderManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) {
        if (page.items.isEmpty) {
          return const Center(child: Text('暂无订单'));
        }
        return RefreshIndicator(
          onRefresh: () => ref.read(orderManageProvider.notifier).refresh(),
          child: LoadMoreListener(
            onLoadMore: () => ref.read(orderManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.receipt_long),
                    title: Text(item.orderNo),
                    subtitle: Text('¥${item.amount} | ${item.statusName}'),
                  ),
                );
              },
            ),
          ),
        );
      },
    );
  }

  Widget _buildRefundList(ThemeData theme, AsyncValue<PagedList<RefundRecordVO>> state) {
    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(extractErrorMessage(e), style: TextStyle(color: theme.colorScheme.error)),
            SizedBox(height: AppTheme.spacingM),
            FilledButton(
              onPressed: () => ref.read(refundManageProvider.notifier).refresh(),
              child: const Text('重试'),
            ),
          ],
        ),
      ),
      data: (page) {
        if (page.items.isEmpty) {
          return const Center(child: Text('暂无退款申请'));
        }
        return RefreshIndicator(
          onRefresh: () => ref.read(refundManageProvider.notifier).refresh(),
          child: LoadMoreListener(
            onLoadMore: () => ref.read(refundManageProvider.notifier).loadMore(),
            child: ListView.builder(
              itemCount: page.items.length,
              itemBuilder: (context, index) {
                final item = page.items[index];
                return Card(
                  child: ListTile(
                    leading: const Icon(Icons.money_off),
                    title: Text('退款单号: ${item.orderNo}'),
                    subtitle: Text('¥${item.amount} | ${item.statusName}'),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: Icon(
                            Icons.check_circle,
                            size: 20,
                            color: AppTheme.successColor,
                          ),
                          tooltip: '通过',
                          onPressed: () => _auditRefund(item.id, true),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.cancel,
                            size: 20,
                            color: AppTheme.errorColor,
                          ),
                          tooltip: '驳回',
                          onPressed: () => _auditRefund(item.id, false),
                        ),
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
