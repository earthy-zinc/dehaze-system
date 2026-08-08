import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/feedback_model.dart';
import '../../providers/providers.dart';
import '../../utils/ui_utils.dart';

/// 反馈评价 — L2 页面
///
/// 我的反馈 + 我的评价，对接 FeedbackService
class FeedbackPage extends ConsumerStatefulWidget {
  const FeedbackPage({super.key});

  @override
  ConsumerState<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends ConsumerState<FeedbackPage>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;

  List<FeedbackPageVO> _feedbacks = [];
  List<MyRatingVO> _ratings = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final service = ref.read(feedbackServiceProvider);
      final feedbackResult = await service.getMyFeedbacks(
        pageNum: 1,
        pageSize: 20,
      );
      final ratingResult = await service.getMyRatings(
        pageNum: 1,
        pageSize: 20,
      );
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _feedbacks = feedbackResult.list;
        _ratings = ratingResult.list;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _submitFeedback() async {
    final titleCtl = TextEditingController();
    final contentCtl = TextEditingController();
    final shouldSubmit = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => Padding(
        padding: EdgeInsets.only(
          left: 16,
          right: 16,
          top: 16,
          bottom: MediaQuery.of(ctx).viewInsets.bottom + 16,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              '提交反馈',
              style: Theme.of(ctx)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: titleCtl,
              decoration: const InputDecoration(
                labelText: '标题',
                hintText: '请输入反馈标题',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: contentCtl,
              maxLines: 4,
              decoration: const InputDecoration(
                labelText: '内容',
                hintText: '请详细描述您的问题或建议',
              ),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('提交'),
            ),
          ],
        ),
      ),
    );

    if (shouldSubmit != true) return;
    if (titleCtl.text.trim().isEmpty || contentCtl.text.trim().isEmpty) {
      if (!mounted) return;
      showError(context, '标题和内容不能为空');
      return;
    }

    try {
      final service = ref.read(feedbackServiceProvider);
      await service.createFeedback(
        FeedbackCreateForm(
          type: FeedbackType.suggestion.value,
          title: titleCtl.text.trim(),
          content: contentCtl.text.trim(),
        ),
      );
      if (!mounted) return;
      showSnackBar(context, '反馈已提交');
      _load();
    } catch (e) {
      if (!mounted) return;
      showError(context, '提交失败: $e');
    }
  }

  void _showFeedbackDetail(FeedbackPageVO fb) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => DraggableScrollableSheet(
        initialChildSize: 0.5,
        minChildSize: 0.3,
        maxChildSize: 0.85,
        expand: false,
        builder: (ctx, scrollController) => Padding(
          padding: const EdgeInsets.all(20),
          child: ListView(
            controller: scrollController,
            children: [
              Text(
                fb.title,
                style: Theme.of(ctx).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
              ),
              const SizedBox(height: 12),
              _detailRow('类型', fb.typeName ?? fb.type),
              _detailRow('状态', fb.statusName ?? fb.status),
              _detailRow('提交时间', fb.createTime),
              if (fb.updateTime != null) _detailRow('更新时间', fb.updateTime!),
            ],
          ),
        ),
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 72,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
            ),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('反馈评价'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: '我的反馈'),
            Tab(text: '我的评价'),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _submitFeedback,
        icon: const Icon(Icons.add),
        label: const Text('提交反馈'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _buildError(theme)
              : TabBarView(
                  controller: _tabController,
                  children: [
                    _buildFeedbackList(theme),
                    _buildRatingList(theme),
                  ],
                ),
    );
  }

  Widget _buildFeedbackList(ThemeData theme) {
    if (_feedbacks.isEmpty) {
      return _buildEmpty(theme, '暂无反馈');
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _feedbacks.length,
      itemBuilder: (context, index) {
        final fb = _feedbacks[index];
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          child: ListTile(
            title: Text(fb.title),
            subtitle: Text(
              '${fb.typeName ?? fb.type}  ·  ${fb.statusName ?? fb.status}  ·  ${fb.createTime}',
            ),
            trailing: const Icon(Icons.chevron_right),
            onTap: () => _showFeedbackDetail(fb),
          ),
        );
      },
    );
  }

  Widget _buildRatingList(ThemeData theme) {
    if (_ratings.isEmpty) {
      return _buildEmpty(theme, '暂无评价');
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _ratings.length,
      itemBuilder: (context, index) {
        final r = _ratings[index];
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
                        r.algorithmName,
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Row(
                      children: List.generate(5, (i) {
                        return Icon(
                          i < r.rating ? Icons.star : Icons.star_border,
                          size: 20,
                          color: i < r.rating
                              ? Colors.amber
                              : theme.colorScheme.onSurfaceVariant,
                        );
                      }),
                    ),
                  ],
                ),
                if (r.comment != null && r.comment!.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    r.comment!,
                    style: theme.textTheme.bodyMedium,
                  ),
                ],
                const SizedBox(height: 8),
                Text(
                  r.createTime,
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildError(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
            const SizedBox(height: 12),
            Text(
              _error!,
              style: theme.textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _load, child: const Text('重试')),
          ],
        ),
      );

  Widget _buildEmpty(ThemeData theme, String message) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.feedback_outlined,
              size: 64,
              color: theme.colorScheme.onSurface.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 16),
            Text(
              message,
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
}
