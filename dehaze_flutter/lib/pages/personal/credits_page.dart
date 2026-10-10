import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/billing_model.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 我的积分 — L2 页面
///
/// 展示用户积分余额与日/月使用限额，对接 BillingService.getBalance。
/// 图像任务按模型价格表以积分计费，会员权益改为月度积分赠送。
class CreditsPage extends ConsumerStatefulWidget {
  const CreditsPage({super.key});

  @override
  ConsumerState<CreditsPage> createState() => _CreditsPageState();
}

class _CreditsPageState extends ConsumerState<CreditsPage> {
  bool _isLoading = true;
  String? _error;
  BalanceVO? _balance;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final balance = await ref.read(billingServiceProvider).getBalance();
      if (!mounted) return;
      setState(() {
        _balance = balance;
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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的积分')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? _buildError(theme)
          : _buildContent(theme),
    );
  }

  Widget _buildContent(ThemeData theme) {
    final balance = _balance!;
    final monthlyUsedRatio = balance.monthlyLimit > 0
        ? balance.monthlyUsed / balance.monthlyLimit
        : 0.0;
    final dailyUsedRatio = balance.dailyLimit > 0
        ? balance.dailyUsed / balance.dailyLimit
        : 0.0;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                children: [
                  Text(
                    '积分余额',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    balance.creditsBalance,
                    style: theme.textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: AppTheme.brandBlue,
                    ),
                  ),
                  if (balance.arrearsStatus) ...[
                    const SizedBox(height: 8),
                    Text(
                      '已欠费，请充值后继续使用',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            '使用情况',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          _buildUsageRow(
            theme,
            '今日已用',
            '${balance.dailyUsed} / ${balance.dailyLimit}',
            dailyUsedRatio,
          ),
          const SizedBox(height: 12),
          _buildUsageRow(
            theme,
            '本月已用',
            '${balance.monthlyUsed} / ${balance.monthlyLimit}',
            monthlyUsedRatio,
          ),
          const SizedBox(height: 16),
          Text(
            '积分说明',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '• 图像任务按所选模型的价格表以积分结算\n'
            '• 缓存命中、拦截、失败与取消的任务不计费\n'
            '• 会员权益以月度积分赠送形式发放',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUsageRow(
    ThemeData theme,
    String label,
    String value,
    double ratio,
  ) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: theme.textTheme.bodyMedium),
          Text(
            value,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
      const SizedBox(height: 8),
      ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: LinearProgressIndicator(
          value: ratio.clamp(0.0, 1.0),
          minHeight: 8,
          backgroundColor: theme.colorScheme.surfaceContainerHighest,
        ),
      ),
    ],
  );

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
}
