import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/evaluation_model.dart';
import '../../models/prediction_model.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import '../../utils/ui_utils.dart';

/// 指标管理页面（L2，ToolsStack 内）
///
/// 查询评估指标历史（ModelService.getEvalMetrics），支持列表+筛选+对比表格。
/// 注意：与 L3 comparison/metrics.dart 的区别 —— 后者是对比页的指标模式（单次评估详情）。
class MetricsManagePage extends ConsumerStatefulWidget {
  const MetricsManagePage({super.key});

  @override
  ConsumerState<MetricsManagePage> createState() => _MetricsManagePageState();
}

class _MetricsManagePageState extends ConsumerState<MetricsManagePage> {
  List<EvaluationResult> _logs = [];
  bool _isLoading = true;
  String? _errorMessage;

  // 筛选
  String? _filterMetric;
  final List<String> _metricNames = ['PSNR', 'SSIM', 'MSE', 'FSIM', 'LPIPS'];

  // 对比选中（最多 5 项）
  final Set<int> _selectedIds = {};

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadLogs());
  }

  Future<void> _loadLogs() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final service = ref.read(evaluationServiceProvider);
      final result = await service.getEvaluationLogs(pageSize: 50);
      if (mounted) {
        setState(() {
          _logs = result.list;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = extractErrorMessage(e);
          _isLoading = false;
        });
      }
    }
  }

  List<EvaluationResult> get _filteredLogs {
    if (_filterMetric == null) return _logs;
    return _logs.where((log) {
      return log.metrics?.containsKey(_filterMetric!.toLowerCase()) == true;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1200,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildHeader(theme)),
            SliverToBoxAdapter(child: _buildFilterBar(theme)),
            SliverPadding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              sliver: _buildContent(theme),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.analytics_outlined,
                    color: AppTheme.indigo, size: 24),
                const SizedBox(width: 8),
                Text(
                  '指标管理',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const Spacer(),
                if (_selectedIds.isNotEmpty)
                  TextButton.icon(
                    onPressed: _clearSelection,
                    icon: const Icon(Icons.clear_all, size: 18),
                    label: const Text('清除选择'),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '查看评估指标历史记录，支持筛选和对比',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );

  Widget _buildFilterBar(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        color: theme.colorScheme.surface,
        child: Row(
          children: [
            Text('指标筛选:',
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                )),
            const SizedBox(width: 12),
            Expanded(
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    FilterChip(
                      label: const Text('全部'),
                      selected: _filterMetric == null,
                      onSelected: (_) =>
                          setState(() => _filterMetric = null),
                    ),
                    const SizedBox(width: 8),
                    ..._metricNames.map((name) => Padding(
                          padding: const EdgeInsets.only(right: 8),
                          child: FilterChip(
                            label: Text(name),
                            selected: _filterMetric == name,
                            onSelected: (selected) => setState(() =>
                                _filterMetric = selected ? name : null),
                          ),
                        )),
                  ],
                ),
              ),
            ),
          ],
        ),
      );

  Widget _buildContent(ThemeData theme) {
    if (_isLoading) {
      return const SliverFillRemaining(
        child: Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return SliverFillRemaining(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, size: 64, color: theme.colorScheme.error),
              const SizedBox(height: 16),
              Text('加载失败', style: theme.textTheme.titleLarge),
              const SizedBox(height: 8),
              Text(_errorMessage!, style: theme.textTheme.bodyMedium),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _loadLogs,
                icon: const Icon(Icons.refresh),
                label: const Text('重试'),
              ),
            ],
          ),
        ),
      );
    }

    if (_logs.isEmpty) {
      return const SliverFillRemaining(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.analytics_outlined, size: 64, color: AppTheme.gray300),
              SizedBox(height: 12),
              Text('暂无评估记录', style: TextStyle(color: AppTheme.gray500)),
            ],
          ),
        ),
      );
    }

    final filtered = _filteredLogs;

    if (filtered.isEmpty) {
      return const SliverFillRemaining(
        child: Center(child: Text('无匹配的评估记录')),
      );
    }

    return SliverToBoxAdapter(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 对比表格（如果有选中项）
          if (_selectedIds.isNotEmpty) ...[
            _buildComparisonTable(theme),
            const SizedBox(height: 16),
          ],
          // 评估记录列表
          Text(
            '评估记录 (${filtered.length})',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          ...filtered.map((log) => _buildLogCard(theme, log)),
        ],
      ),
    );
  }

  Widget _buildLogCard(ThemeData theme, EvaluationResult log) {
    final isSelected = _selectedIds.contains(log.logId);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        side: isSelected
            ? BorderSide(color: AppTheme.brandBlue, width: 2)
            : BorderSide.none,
      ),
      child: InkWell(
        onTap: () => _toggleSelection(log.logId!),
        borderRadius: BorderRadius.circular(AppTheme.radiusM),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _statusBadge(theme, log.status),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      '评估记录 #${log.logId}',
                      style: theme.textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  if (isSelected)
                    Icon(Icons.check_circle,
                        color: AppTheme.brandBlue, size: 20),
                ],
              ),
              if (log.metrics != null && log.metrics!.isNotEmpty) ...[
                const SizedBox(height: 12),
                _buildMetricsRow(theme, log.metrics!),
              ],
              if (log.time != null) ...[
                const SizedBox(height: 8),
                Text(
                  '耗时: ${(log.time! / 1000).toStringAsFixed(1)}s',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
              if (log.errorMessage != null) ...[
                const SizedBox(height: 4),
                Text(
                  log.errorMessage!,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: AppTheme.errorColor,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _statusBadge(ThemeData theme, TaskStatus status) {
    final (color, text) = switch (status) {
      TaskStatus.completed => (AppTheme.techGreen, '完成'),
      TaskStatus.failed => (AppTheme.errorColor, '失败'),
      TaskStatus.processing => (AppTheme.warningColor, '处理中'),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w500,
          color: color,
        ),
      ),
    );
  }

  Widget _buildMetricsRow(ThemeData theme, Map<String, double> metrics) {
    final model = EvaluationMetrics.fromMap(metrics);
    final items = model.toList().where((m) => m.value != null).toList();

    return Wrap(
      spacing: 16,
      runSpacing: 8,
      children: items.map((item) {
        final color = item.higherIsBetter ? AppTheme.techGreen : AppTheme.errorColor;
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              item.name,
              style: theme.textTheme.bodySmall?.copyWith(
                fontWeight: FontWeight.w500,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(width: 4),
            Text(
              item.displayValue,
              style: theme.textTheme.bodySmall?.copyWith(
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        );
      }).toList(),
    );
  }

  Widget _buildComparisonTable(ThemeData theme) {
    final selectedLogs = _logs
        .where((log) => _selectedIds.contains(log.logId) && log.metrics != null)
        .toList();

    if (selectedLogs.isEmpty) return const SizedBox.shrink();

    // 收集所有指标名称
    final allMetricKeys = <String>{};
    for (final log in selectedLogs) {
      allMetricKeys.addAll(log.metrics!.keys.map((k) => k.toUpperCase()));
    }
    final metricKeys = allMetricKeys.toList()..sort();

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.colorScheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              '指标对比 (${selectedLogs.length} 项)',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: DataTable(
              columns: [
                const DataColumn(label: Text('指标')),
                ...selectedLogs.map((log) => DataColumn(
                      label: Text('#${log.logId}',
                          style: const TextStyle(fontWeight: FontWeight.w600)),
                    )),
              ],
              rows: metricKeys.map((key) {
                final metricItem = EvaluationMetrics.fromMap(
                  {key.toLowerCase(): 0.0},
                ).toList().firstWhere(
                  (m) => m.name == key,
                  orElse: () => const MetricItem(
                    name: '',
                    value: null,
                    unit: '',
                    higherIsBetter: true,
                    description: '',
                  ),
                );

                return DataRow(cells: [
                  DataCell(Text(key)),
                  ...selectedLogs.map((log) {
                    final value = log.metrics![key.toLowerCase()];
                    final color = metricItem.higherIsBetter
                        ? AppTheme.techGreen
                        : AppTheme.errorColor;
                    return DataCell(Text(
                      value != null
                          ? value.toStringAsFixed(value < 1 ? 4 : 2)
                          : '-',
                      style: TextStyle(
                        fontWeight: FontWeight.w500,
                        color: value != null ? color : null,
                      ),
                    ));
                  }),
                ]);
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  void _toggleSelection(int logId) {
    setState(() {
      if (_selectedIds.contains(logId)) {
        _selectedIds.remove(logId);
      } else {
        if (_selectedIds.length >= 5) {
          showSnackBar(context, '最多选择 5 项进行对比');
          return;
        }
        _selectedIds.add(logId);
      }
    });
  }

  void _clearSelection() {
    setState(() => _selectedIds.clear());
  }
}
