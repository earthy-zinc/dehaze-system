import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/package_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/format_utils.dart';

/// 我的套餐 — L2 页面
///
/// 展示可选套餐列表，对接 PackageService.getList
class PackagePage extends ConsumerStatefulWidget {
  const PackagePage({super.key});

  @override
  ConsumerState<PackagePage> createState() => _PackagePageState();
}

class _PackagePageState extends ConsumerState<PackagePage> {
  bool _isLoading = true;
  String? _error;
  List<PackageDetailVO> _packages = [];

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
      final list = await packageService.getList();
      if (!mounted) return;
      setState(() {
        _packages = list;
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

  void _onBuyPackage(PackageDetailVO pkg) {
    context.push(AppRouterConfig.orderCreate, extra: pkg);
  }

  void _onViewDetail(PackageDetailVO pkg) {
    context.push(
      AppRouterConfig.packageDetail.replaceFirst(':id', '${pkg.id}'),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的套餐')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? _buildError(theme)
          : _packages.isEmpty
          ? _buildEmpty(theme)
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _packages.length,
              itemBuilder: (context, index) {
                final pkg = _packages[index];
                return _buildPackageCard(theme, pkg);
              },
            ),
    );
  }

  Widget _buildPackageCard(ThemeData theme, PackageDetailVO pkg) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
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
              const SizedBox(height: 8),
              Text(
                pkg.description!,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
            if (pkg.benefits.isNotEmpty) ...[
              const SizedBox(height: 8),
              ...pkg.benefits.entries.map(
                (e) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
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
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    if (pkg.originalPrice > pkg.salePrice)
                      Padding(
                        padding: const EdgeInsets.only(right: 6),
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
                      style: theme.textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                        color: AppTheme.errorColor,
                      ),
                    ),
                  ],
                ),
                Row(
                  children: [
                    TextButton(
                      onPressed: () => _onViewDetail(pkg),
                      child: const Text('详情'),
                    ),
                    const SizedBox(width: 4),
                    FilledButton(
                      onPressed: () => _onBuyPackage(pkg),
                      child: const Text('立即购买'),
                    ),
                  ],
                ),
              ],
            ),
          ],
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
        Icon(
          Icons.inventory_2_outlined,
          size: 64,
          color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
        ),
        const SizedBox(height: 16),
        Text(
          '暂无可用套餐',
          style: theme.textTheme.titleMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    ),
  );
}
