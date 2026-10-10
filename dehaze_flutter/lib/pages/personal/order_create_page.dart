import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/network/api_result.dart';
import '../../models/order_model.dart';
import '../../models/package_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/format_utils.dart';
import '../../utils/ui_utils.dart';

/// 确认下单 — L2 页面
///
/// 进入即预览价格明细，确认后以余额支付创建订单并跳转我的订单。
class OrderCreatePage extends ConsumerStatefulWidget {
  const OrderCreatePage({super.key, required this.package});

  final PackageDetailVO package;

  @override
  ConsumerState<OrderCreatePage> createState() => _OrderCreatePageState();
}

class _OrderCreatePageState extends ConsumerState<OrderCreatePage> {
  bool _isLoading = true;
  bool _isPaying = false;
  String? _error;
  PriceResult? _price;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadPrice());
  }

  Future<void> _loadPrice() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final packageService = ref.read(packageServiceProvider);
      final price = await packageService.calculatePrice(widget.package.id);
      if (!mounted) return;
      setState(() {
        _price = price;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _confirmPay() async {
    if (_isPaying) return;
    setState(() => _isPaying = true);
    try {
      final orderService = ref.read(orderServiceProvider);
      final result = await orderService.create(
        OrderCreateForm(
          packageId: widget.package.id,
          payMethod: PayMethod.balance,
        ),
      );
      if (!mounted) return;
      showSnackBar(context, result.paid ? '支付成功' : '订单已创建，请在订单页完成支付');
      context.go(AppRouterConfig.orders);
    } catch (e) {
      if (!mounted) return;
      setState(() => _isPaying = false);
      showError(context, extractErrorMessage(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('确认订单')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? _buildError(theme)
          : _buildContent(theme, _price!),
    );
  }

  Widget _buildContent(ThemeData theme, PriceResult price) {
    return Column(
      children: [
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _buildPackageCard(theme),
              const SizedBox(height: 12),
              _buildPriceCard(theme, price),
              const SizedBox(height: 12),
              _buildPayMethodCard(theme),
            ],
          ),
        ),
        _buildPayBar(),
      ],
    );
  }

  Widget _buildPackageCard(ThemeData theme) {
    final pkg = widget.package;
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              pkg.name,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              pkg.levelName != null
                  ? '${pkg.levelName} · ${packageSpecLabel(pkg)}'
                  : packageSpecLabel(pkg),
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPriceCard(ThemeData theme, PriceResult price) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _priceRow(theme, '原价', formatYuan(price.originalPrice)),
            _priceRow(
              theme,
              '促销优惠',
              '-${formatYuan(price.discountAmount)}',
              valueColor: AppTheme.techGreen,
            ),
            if (price.couponAmount > 0)
              _priceRow(
                theme,
                '优惠券抵扣',
                '-${formatYuan(price.couponAmount)}',
                valueColor: AppTheme.techGreen,
              ),
            const Divider(height: 24),
            _priceRow(
              theme,
              '应付',
              formatYuan(price.payableAmount),
              valueColor: AppTheme.errorColor,
              emphasize: true,
            ),
          ],
        ),
      ),
    );
  }

  Widget _priceRow(
    ThemeData theme,
    String label,
    String value, {
    Color? valueColor,
    bool emphasize = false,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
          Text(
            value,
            style: emphasize
                ? theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: valueColor,
                  )
                : theme.textTheme.bodyMedium?.copyWith(color: valueColor),
          ),
        ],
      ),
    );
  }

  Widget _buildPayMethodCard(ThemeData theme) {
    return Card(
      margin: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            child: Text(
              '支付方式',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const ListTile(
            leading: Icon(Icons.account_balance_wallet_outlined),
            title: Text('余额支付'),
            trailing: Icon(Icons.check_circle, color: AppTheme.techGreen),
          ),
        ],
      ),
    );
  }

  Widget _buildPayBar() {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SizedBox(
          width: double.infinity,
          child: FilledButton(
            onPressed: _isPaying ? null : _confirmPay,
            child: _isPaying
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('确认支付'),
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
        ElevatedButton(onPressed: _loadPrice, child: const Text('重试')),
      ],
    ),
  );
}
