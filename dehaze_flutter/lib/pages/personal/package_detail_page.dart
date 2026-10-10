import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/package_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/format_utils.dart';

/// 套餐详情 — L2 页面
///
/// 对接 PackageService.getDetail，展示套餐完整信息（含进行中促销），底部进入下单页。
class PackageDetailPage extends ConsumerStatefulWidget {
  const PackageDetailPage({super.key, required this.packageId});

  final int packageId;

  @override
  ConsumerState<PackageDetailPage> createState() => _PackageDetailPageState();
}

class _PackageDetailPageState extends ConsumerState<PackageDetailPage> {
  bool _isLoading = true;
  String? _error;
  PackageDetailVO? _package;

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
      final packageService = ref.read(packageServiceProvider);
      final detail = await packageService.getDetail(widget.packageId);
      if (!mounted) return;
      setState(() {
        _package = detail;
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
      appBar: AppBar(title: const Text('套餐详情')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? _buildError(theme)
          : _buildContent(theme, _package!),
    );
  }

  Widget _buildContent(ThemeData theme, PackageDetailVO pkg) {
    return Column(
      children: [
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              _buildHeaderCard(theme, pkg),
              if (pkg.benefits.isNotEmpty) ...[
                const SizedBox(height: 12),
                _buildBenefitsCard(theme, pkg),
              ],
              const SizedBox(height: 12),
              _buildPriceCard(theme, pkg),
            ],
          ),
        ),
        _buildBuyBar(pkg),
      ],
    );
  }

  Widget _buildHeaderCard(ThemeData theme, PackageDetailVO pkg) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    pkg.name,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: AppTheme.brandBlue.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    pkg.levelName ?? packageTypeLabel(pkg.packageType),
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: AppTheme.brandBlue,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              packageSpecLabel(pkg),
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (pkg.description != null && pkg.description!.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text(
                pkg.description!,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildBenefitsCard(ThemeData theme, PackageDetailVO pkg) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '套餐权益',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            ...pkg.benefits.entries.map(
              (e) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    const Icon(
                      Icons.check_circle_outline,
                      size: 16,
                      color: AppTheme.techGreen,
                    ),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        '${packageBenefitLabels[e.key] ?? e.key}：${e.value}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPriceCard(ThemeData theme, PackageDetailVO pkg) {
    return Card(
      margin: EdgeInsets.zero,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            if (pkg.originalPrice > pkg.salePrice)
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Text(
                  formatYuan(pkg.originalPrice),
                  style: theme.textTheme.bodySmall?.copyWith(
                    decoration: TextDecoration.lineThrough,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            Text(
              formatYuan(pkg.salePrice),
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w700,
                color: AppTheme.errorColor,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBuyBar(PackageDetailVO pkg) {
    return SafeArea(
      top: false,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SizedBox(
          width: double.infinity,
          child: FilledButton(
            onPressed: () =>
                context.push(AppRouterConfig.orderCreate, extra: pkg),
            child: const Text('立即购买'),
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
}
