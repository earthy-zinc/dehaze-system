import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/prediction_model.dart';
import '../../providers/processing_provider.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

/// 处理历史页面
class TaskHistoryPage extends ConsumerStatefulWidget {
  const TaskHistoryPage({super.key});

  @override
  ConsumerState<TaskHistoryPage> createState() => _TaskHistoryPageState();
}

class _TaskHistoryPageState extends ConsumerState<TaskHistoryPage> {
  List<PredictionLog> _logs = [];
  bool _isLoading = true;
  String? _errorMessage;
  int _currentPage = 1;
  int _total = 0;
  static const int _pageSize = 20;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadLogs());
  }

  Future<void> _loadLogs({bool refresh = false}) async {
    if (refresh) {
      _currentPage = 1;
      setState(() => _isLoading = true);
    }

    try {
      final service = ref.read(predictionServiceProvider);
      final result = await service.getPredictionLogs(
        pageNum: _currentPage,
        pageSize: _pageSize,
      );

      setState(() {
        if (refresh) {
          _logs = result.list;
        } else {
          _logs.addAll(result.list);
        }
        _total = result.total;
        _isLoading = false;
        _errorMessage = null;
      });
    } catch (e) {
      setState(() {
        _errorMessage = _extractError(e);
        _isLoading = false;
      });
    }
  }

  String _extractError(dynamic e) {
    if (e is ApiException) return e.message;
    return e.toString().replaceFirst('Exception: ', '');
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1000,
        child: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildHeader(theme)),
            if (_isLoading && _logs.isEmpty)
              const SliverFillRemaining(child: Center(child: CircularProgressIndicator()))
            else if (_errorMessage != null)
              SliverFillRemaining(child: _buildError(theme))
            else if (_logs.isEmpty)
              SliverFillRemaining(child: _buildEmpty(theme))
            else
              SliverPadding(
                padding: const EdgeInsets.all(16),
                sliver: SliverList(
                  delegate: SliverChildBuilderDelegate(
                    (context, index) => _LogCard(log: _logs[index]),
                    childCount: _logs.length,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(bottom: BorderSide(color: theme.dividerColor)),
        ),
        child: Row(
          children: [
            Icon(Icons.history, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('处理历史', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const Spacer(),
            if (_total > 0)
              Text('共 $_total 条', style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );

  Widget _buildEmpty(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.inbox_outlined, size: 64, color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
            const SizedBox(height: 16),
            Text('暂无处理记录', style: theme.textTheme.titleMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );

  Widget _buildError(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 64, color: theme.colorScheme.error),
            const SizedBox(height: 16),
            Text(_errorMessage!, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () => _loadLogs(refresh: true),
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );
}

/// 日志卡片
class _LogCard extends StatelessWidget {
  const _LogCard({required this.log});
  final PredictionLog log;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        onTap: () => _showDetail(context),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: log.predUrl != null
                    ? Image.network(
                        log.predUrl!,
                        width: 80,
                        height: 80,
                        fit: BoxFit.cover,
                        errorBuilder: (_, _, _) => Container(
                          width: 80,
                          height: 80,
                          color: theme.colorScheme.surfaceContainerHighest,
                          child: const Icon(Icons.broken_image, size: 32),
                        ),
                      )
                    : Container(
                        width: 80,
                        height: 80,
                        color: theme.colorScheme.surfaceContainerHighest,
                        child: const Icon(Icons.image_not_supported, size: 32),
                      ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(log.algorithmName,
                        style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w600),
                        maxLines: 1, overflow: TextOverflow.ellipsis),
                    const SizedBox(height: 4),
                    Text(log.createTime,
                        style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                    if (log.time != null) ...[
                      const SizedBox(height: 4),
                      Text('耗时 ${(log.time! / 1000).toStringAsFixed(1)}s',
                          style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                    ],
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: theme.colorScheme.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }

  void _showDetail(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (ctx) => Dialog(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800, maxHeight: 600),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AppBar(
                title: Text(log.algorithmName),
                leading: IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.of(ctx).pop(),
                ),
              ),
              Expanded(
                child: Row(
                  children: [
                    Expanded(child: _imageBlock('原图', log.originUrl)),
                    Expanded(child: _imageBlock('结果', log.predUrl)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _imageBlock(String label, String? url) => Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
          ),
          Expanded(
            child: url != null
                ? Image.network(url, fit: BoxFit.contain,
                    errorBuilder: (_, _, _) => const Center(child: Icon(Icons.broken_image)))
                : const Center(child: Icon(Icons.image_not_supported)),
          ),
        ],
      );
}
