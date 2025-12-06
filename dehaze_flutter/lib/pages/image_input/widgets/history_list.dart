import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/image_input_model.dart';
import '../providers/history_provider.dart';
import '../providers/image_input_provider.dart';
import 'history_item_card.dart';

/// 历史记录列表
///
/// 显示分组的历史记录，支持删除和重新处理
class HistoryList extends ConsumerWidget {
  const HistoryList({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final groupedHistory = ref.watch(groupedHistoryProvider);
    final historyCount = ref.watch(historyCountProvider);
    final theme = Theme.of(context);

    if (historyCount == 0) {
      return _buildEmptyState(theme);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 头部
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: [
              Text(
                '最近处理的图片',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const Spacer(),
              TextButton.icon(
                onPressed: () => _showClearConfirmDialog(context, ref),
                icon: Icon(
                  Icons.delete_sweep_outlined,
                  size: 18,
                  color: theme.colorScheme.error,
                ),
                label: Text(
                  '清空',
                  style: TextStyle(color: theme.colorScheme.error),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 8),

        // 列表
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: _getTotalItemCount(groupedHistory),
            itemBuilder: (context, index) {
              return _buildItem(context, ref, groupedHistory, index);
            },
          ),
        ),
      ],
    );
  }

  Widget _buildEmptyState(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.inbox_outlined,
              size: 64,
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.5),
            ),
            const SizedBox(height: 16),
            Text(
              '暂无历史记录',
              style: theme.textTheme.titleMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '处理过的图片会显示在这里',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.7),
              ),
            ),
          ],
        ),
      );

  int _getTotalItemCount(Map<String, List<HistoryRecordModel>> grouped) {
    var count = 0;
    for (final entry in grouped.entries) {
      count += 1; // 分组标题
      count += entry.value.length; // 记录数
    }
    return count;
  }

  Widget _buildItem(
    BuildContext context,
    WidgetRef ref,
    Map<String, List<HistoryRecordModel>> grouped,
    int index,
  ) {
    var currentIndex = 0;

    for (final entry in grouped.entries) {
      // 分组标题
      if (currentIndex == index) {
        return _buildGroupHeader(context, entry.key);
      }
      currentIndex++;

      // 分组内的记录
      for (final record in entry.value) {
        if (currentIndex == index) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: HistoryItemCard(
              record: record,
              onTap: () => _loadRecord(ref, record),
              onReprocess: () => _loadRecord(ref, record),
              onDelete: () => _deleteRecord(ref, record.id),
            ),
          );
        }
        currentIndex++;
      }
    }

    return const SizedBox.shrink();
  }

  Widget _buildGroupHeader(BuildContext context, String title) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.only(top: 16, bottom: 8),
      child: Text(
        title,
        style: theme.textTheme.labelLarge?.copyWith(
          fontWeight: FontWeight.w600,
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }

  void _loadRecord(WidgetRef ref, HistoryRecordModel record) {
    ref.read(imageInputProvider.notifier).loadFromHistory(record);
  }

  void _deleteRecord(WidgetRef ref, String id) {
    ref.read(historyProvider.notifier).deleteRecord(id);
  }

  void _showClearConfirmDialog(BuildContext context, WidgetRef ref) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('清空历史记录'),
        content: const Text('确定要清空所有历史记录吗？此操作无法撤销。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('取消'),
          ),
          FilledButton(
            onPressed: () {
              ref.read(historyProvider.notifier).clearAll();
              Navigator.of(context).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('历史记录已清空')),
              );
            },
            style: FilledButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.error,
            ),
            child: const Text('清空'),
          ),
        ],
      ),
    );
  }
}
