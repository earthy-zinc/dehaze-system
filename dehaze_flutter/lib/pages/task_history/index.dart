import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/network/api_result.dart';
import '../../models/prediction_model.dart';
import '../../providers/processing_provider.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import '../../utils/ui_utils.dart';
import '../../widgets/dehaze_image.dart';

/// 处理历史页面
class TaskHistoryPage extends ConsumerStatefulWidget {
  const TaskHistoryPage({super.key});

  @override
  ConsumerState<TaskHistoryPage> createState() => _TaskHistoryPageState();
}

class _TaskHistoryPageState extends ConsumerState<TaskHistoryPage> {
  List<PredictionLog> _logs = [];
  bool _isLoading = true;
  bool _isLoadingMore = false;
  String? _errorMessage;
  int _currentPage = 1;
  int _total = 0;
  static const int _pageSize = 20;

  /// 是否还有更多数据可加载
  bool get _hasMore => _logs.length < _total;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadLogs(refresh: true));
  }

  Future<void> _loadLogs({bool refresh = false}) async {
    if (refresh) {
      _currentPage = 1;
      setState(() {
        _isLoading = true;
        _errorMessage = null;
        _logs = [];
      });
    } else {
      // 分页加载：避免重复触发/已加载完毕时直接返回
      if (_isLoading || _isLoadingMore || !_hasMore) return;
      setState(() => _isLoadingMore = true);
      _currentPage += 1;
    }

    try {
      final service = ref.read(predictionServiceProvider);
      final result = await service.getPredictionLogs(
        pageNum: _currentPage,
        pageSize: _pageSize,
      );

      if (!mounted) return;
      setState(() {
        if (refresh) {
          _logs = result.list;
        } else {
          _logs.addAll(result.list);
        }
        _total = result.total;
        _isLoading = false;
        _isLoadingMore = false;
        _errorMessage = null;
      });
    } catch (e) {
      if (!mounted) return;
      if (refresh) {
        setState(() {
          _errorMessage = extractErrorMessage(e);
          _isLoading = false;
        });
      } else {
        // 分页失败：回退页码并提示，保留已加载列表
        setState(() {
          _currentPage -= 1;
          _isLoadingMore = false;
        });
        showSnackBar(context, '加载更多失败: ${extractErrorMessage(e)}');
      }
    }
  }

  /// 滚动到底部附近触发加载下一页
  bool _onScroll(ScrollNotification notification) {
    if (notification is! ScrollEndNotification) return false;
    if (_isLoading || _isLoadingMore || !_hasMore) return false;
    if (notification.metrics.extentAfter < 200) {
      _loadLogs();
    }
    return false;
  }

  /// 下拉刷新：重置页码并重新加载第一页
  ///
  /// 保留当前列表避免刷新时闪烁，仅在新数据到达后替换；
  /// 失败时以 SnackBar 提示，不影响已加载内容。
  Future<void> _refresh() async {
    _currentPage = 1;
    try {
      final service = ref.read(predictionServiceProvider);
      final result = await service.getPredictionLogs(
        pageNum: _currentPage,
        pageSize: _pageSize,
      );
      if (!mounted) return;
      setState(() {
        _logs = result.list;
        _total = result.total;
        _errorMessage = null;
      });
    } catch (e) {
      if (!mounted) return;
      showSnackBar(context, '刷新失败: ${extractErrorMessage(e)}');
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1000,
        child: RefreshIndicator(
          onRefresh: _refresh,
          child: NotificationListener<ScrollNotification>(
            onNotification: _onScroll,
            child: CustomScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              slivers: [
                SliverToBoxAdapter(child: _buildHeader(theme)),
                if (_isLoading && _logs.isEmpty)
                  const SliverFillRemaining(
                      child: Center(child: CircularProgressIndicator()))
                else if (_errorMessage != null)
                  SliverFillRemaining(child: _buildError(theme))
                else if (_logs.isEmpty)
                  SliverFillRemaining(child: _buildEmpty(theme))
                else ...[
                  SliverPadding(
                    padding: const EdgeInsets.all(16),
                    sliver: SliverList(
                      delegate: SliverChildBuilderDelegate(
                        (context, index) => _LogCard(log: _logs[index]),
                        childCount: _logs.length,
                      ),
                    ),
                  ),
                  // 加载更多指示器
                  if (_isLoadingMore)
                    const SliverToBoxAdapter(
                      child: Padding(
                        padding: EdgeInsets.all(16),
                        child: Center(child: CircularProgressIndicator()),
                      ),
                    ),
                  // 已加载完毕提示
                  if (!_hasMore && _logs.isNotEmpty)
                    SliverToBoxAdapter(
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: Center(
                          child: Text(
                            '没有更多了',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ],
            ),
          ),
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
                child: DehazeImage(
                  url: log.predUrl,
                  width: 80,
                  height: 80,
                  fit: BoxFit.cover,
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
            child: DehazeImage(
              url: url,
              fit: BoxFit.contain,
            ),
          ),
        ],
      );
}
