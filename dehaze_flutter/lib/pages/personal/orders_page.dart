import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/order_model.dart';
import '../../providers/providers.dart';

/// 我的订单 — L2 页面
///
/// 对接 OrderService.getMyOrders / getMyOrderDetail / cancelOrder 真实 API。
class OrdersPage extends ConsumerStatefulWidget {
  const OrdersPage({super.key});

  @override
  ConsumerState<OrdersPage> createState() => _OrdersPageState();
}

class _OrdersPageState extends ConsumerState<OrdersPage> {
  final List<MyOrderVO> _items = [];
  bool _isLoading = true;
  String? _error;
  int _pageNum = 1;
  int _total = 0;
  bool _isLoadingMore = false;
  static const int _pageSize = 20;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
      _pageNum = 1;
    });
    try {
      final service = ref.read(orderServiceProvider);
      final result = await service.getMyOrders(
        const MyOrderQuery(pageNum: 1, pageSize: _pageSize),
      );
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _items
          ..clear()
          ..addAll(result.list);
        _total = result.total;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _loadMore() async {
    if (_isLoadingMore || _items.length >= _total) return;
    setState(() => _isLoadingMore = true);
    try {
      final service = ref.read(orderServiceProvider);
      final nextPage = _pageNum + 1;
      final result = await service.getMyOrders(
        MyOrderQuery(pageNum: nextPage, pageSize: _pageSize),
      );
      if (!mounted) return;
      setState(() {
        _isLoadingMore = false;
        _items.addAll(result.list);
        _pageNum = nextPage;
        _total = result.total;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _isLoadingMore = false);
    }
  }

  Future<void> _viewDetail(MyOrderVO item) async {
    try {
      final detail = await ref.read(orderServiceProvider).getMyOrderDetail(
            item.id,
          );
      if (!mounted) return;
      _showDetailDialog(detail);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('获取订单详情失败: $e')),
      );
    }
  }

  Future<void> _cancelOrder(MyOrderVO item) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('确认取消'),
        content: Text('确定取消订单 ${item.orderNo} 吗？'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('返回'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('确定取消'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    try {
      await ref.read(orderServiceProvider).cancelOrder(item.id);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('订单已取消')),
      );
      _load();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('取消订单失败: $e')),
      );
    }
  }

  void _showDetailDialog(OrderDetailVO detail) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('订单详情'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              _detailRow('订单编号', detail.orderNo),
              _detailRow('套餐名称', detail.packageName),
              _detailRow('套餐等级', detail.packageLevel),
              _detailRow('时长', '${detail.period}天'),
              _detailRow('原价', '¥${detail.originalPrice.toStringAsFixed(2)}'),
              _detailRow('折扣', '¥${detail.discount.toStringAsFixed(2)}'),
              _detailRow('实付', '¥${detail.amount.toStringAsFixed(2)}'),
              _detailRow('状态', detail.statusName),
              _detailRow('创建时间', detail.createTime),
              if (detail.payTime != null) _detailRow('支付时间', detail.payTime!),
              if (detail.expireTime != null)
                _detailRow('过期时间', detail.expireTime!),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('关闭'),
          ),
        ],
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 72,
            child: Text(
              '$label:',
              style: const TextStyle(color: Colors.grey),
            ),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  Color _statusColor(int status) {
    switch (status) {
      case 1: // paid
        return Colors.green;
      case 2: // cancelled
        return Colors.grey;
      case 3: // refunded
        return Colors.orange;
      case 4: // expired
        return Colors.brown;
      case 5: // failed
        return Colors.red;
      default: // pending
        return Colors.blue;
    }
  }

  String _formatTime(String time) {
    if (time.length >= 16) return time.substring(0, 16).replaceFirst('T', ' ');
    return time;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的订单')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _buildError(theme)
              : _items.isEmpty
                  ? _buildEmpty(theme)
                  : RefreshIndicator(
                      onRefresh: _load,
                      child: NotificationListener<ScrollNotification>(
                        onNotification: (notification) {
                          if (notification is ScrollEndNotification &&
                              notification.metrics.pixels >=
                                  notification.metrics.maxScrollExtent - 100) {
                            _loadMore();
                          }
                          return false;
                        },
                        child: ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _items.length + (_hasMore ? 1 : 0),
                          itemBuilder: (context, index) {
                            if (index >= _items.length) {
                              return const Padding(
                                padding: EdgeInsets.symmetric(vertical: 16),
                                child:
                                    Center(child: CircularProgressIndicator()),
                              );
                            }
                            final item = _items[index];
                            return _buildOrderCard(item, theme);
                          },
                        ),
                      ),
                    ),
    );
  }

  bool get _hasMore => _items.length < _total;

  Widget _buildOrderCard(MyOrderVO item, ThemeData theme) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () => _viewDetail(item),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      item.packageName,
                      style: theme.textTheme.titleSmall,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: _statusColor(item.status).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      item.statusName,
                      style: TextStyle(
                        color: _statusColor(item.status),
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                '订单号: ${item.orderNo}',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '¥${item.amount.toStringAsFixed(2)}',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: theme.colorScheme.primary,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    _formatTime(item.createTime),
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
              if (item.status == 0) ...[
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton(
                    onPressed: () => _cancelOrder(item),
                    child: const Text(
                      '取消订单',
                      style: TextStyle(color: Colors.red),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildError(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
            const SizedBox(height: 12),
            Text(_error!, style: theme.textTheme.bodyMedium),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _load, child: const Text('重试')),
          ],
        ),
      );

  Widget _buildEmpty(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.receipt_long,
                size: 64,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
            const SizedBox(height: 16),
            Text('暂无订单',
                style: theme.textTheme.titleMedium
                    ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );
}
