import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/order_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../services/order_service.dart';
import '../../theme/app_theme.dart';

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
  List<OrderPageVO> _orders = [];
  List<RefundRecordVO> _refunds = [];
  int _orderTotal = 0, _refundTotal = 0;
  int _orderPageNum = 1, _refundPageNum = 1;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 2, vsync: this);
    _tabCtrl.addListener(() {
      if (!_tabCtrl.indexIsChanging) {
        _fetchCurrent();
      }
    });
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchOrders());
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    _searchController.dispose();
    super.dispose();
  }

  OrderService get _service => ref.read(orderServiceProvider);

  void _fetchCurrent() {
    if (_tabCtrl.index == 0) {
      _fetchOrders();
    } else {
      _fetchRefunds();
    }
  }

  Future<void> _fetchOrders({bool reset = false}) async {
    if (reset) {
      _orderPageNum = 1;
    }
    setState(() {
      _loading = true;
    });
    try {
      final result = await _service.getPage(
        OrderQuery(
          pageNum: _orderPageNum,
          pageSize: 10,
          keyword: _searchController.text,
        ),
      );
      setState(() {
        if (reset) {
          _orders = result.list;
        } else {
          _orders.addAll(result.list);
        }
        _orderTotal = result.total;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
      });
    }
  }

  Future<void> _fetchRefunds({bool reset = false}) async {
    if (reset) {
      _refundPageNum = 1;
    }
    setState(() {
      _loading = true;
    });
    try {
      final result = await _service.getRefundPage(
        RefundQuery(pageNum: _refundPageNum, pageSize: 10),
      );
      setState(() {
        if (reset) {
          _refunds = result.list;
        } else {
          _refunds.addAll(result.list);
        }
        _refundTotal = result.total;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
      });
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
      await _service.auditRefund(
        RefundAuditForm(
          refundId: refundId,
          approved: approved,
          remark: reasonCtrl.text.trim(),
        ),
      );
      _showSnack('审核完成');
      _fetchRefunds(reset: true);
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
    final auth = ref.watch(authProvider);
    if (!auth.hasPerm('sys:order:*')) {
      return Scaffold(
        appBar: AppBar(title: const Text('订单管理')),
        body: const Center(child: Text('无权限访问')),
      );
    }
    final theme = Theme.of(context);

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
                    _fetchCurrent();
                  },
                ),
              ),
              onSubmitted: (_) => _fetchCurrent(),
            ),
          ),
          Expanded(
            child: TabBarView(
              controller: _tabCtrl,
              children: [
                _buildOrderList(theme),
                _buildRefundList(theme),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOrderList(ThemeData theme) {
    if (_loading && _orders.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_orders.isEmpty) {
      return const Center(child: Text('暂无订单'));
    }
    return RefreshIndicator(
      onRefresh: () => _fetchOrders(reset: true),
      child: ListView.builder(
        itemCount: _orders.length + (_orders.length < _orderTotal ? 1 : 0),
        itemBuilder: (context, index) {
          if (index >= _orders.length) {
            if (!_loading) {
              _orderPageNum++;
              _fetchOrders();
            }
            return const Center(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              ),
            );
          }
          final item = _orders[index];
          return Card(
            child: ListTile(
              leading: const Icon(Icons.receipt_long),
              title: Text(item.orderNo),
              subtitle: Text('¥${item.amount} | ${item.statusName}'),
            ),
          );
        },
      ),
    );
  }

  Widget _buildRefundList(ThemeData theme) {
    if (_loading && _refunds.isEmpty) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_refunds.isEmpty) {
      return const Center(child: Text('暂无退款申请'));
    }
    return RefreshIndicator(
      onRefresh: () => _fetchRefunds(reset: true),
      child: ListView.builder(
        itemCount:
            _refunds.length + (_refunds.length < _refundTotal ? 1 : 0),
        itemBuilder: (context, index) {
          if (index >= _refunds.length) {
            if (!_loading) {
              _refundPageNum++;
              _fetchRefunds();
            }
            return const Center(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              ),
            );
          }
          final item = _refunds[index];
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
    );
  }
}
