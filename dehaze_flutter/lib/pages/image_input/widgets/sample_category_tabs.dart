import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/image_input_model.dart';
import '../providers/sample_provider.dart';

/// 样例分类标签
///
/// 水平滚动的分类筛选标签
class SampleCategoryTabs extends ConsumerWidget {
  const SampleCategoryTabs({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedCategory = ref.watch(sampleCategoryProvider);
    final counts = ref.watch(sampleCountsProvider);

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: SampleCategory.values.map((category) {
          final count = counts[category] ?? 0;
          final isSelected = selectedCategory == category;

          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: _CategoryChip(
              label: category.displayName,
              count: count,
              isSelected: isSelected,
              onTap: () {
                ref.read(sampleProvider.notifier).filterByCategory(category);
              },
            ),
          );
        }).toList(),
      ),
    );
  }
}

/// 分类标签芯片
class _CategoryChip extends StatefulWidget {
  const _CategoryChip({
    required this.label,
    required this.count,
    required this.isSelected,
    required this.onTap,
  });

  final String label;
  final int count;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  State<_CategoryChip> createState() => _CategoryChipState();
}

class _CategoryChipState extends State<_CategoryChip> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: widget.isSelected
                ? const Color(0xFF3B82F6)
                : _isHovered
                    ? theme.colorScheme.surfaceContainerHighest
                    : theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: widget.isSelected
                  ? const Color(0xFF3B82F6)
                  : theme.dividerColor,
              width: 1,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                widget.label,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: widget.isSelected ? FontWeight.w600 : FontWeight.w500,
                  color: widget.isSelected ? Colors.white : theme.colorScheme.onSurface,
                ),
              ),
              if (widget.count > 0) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: widget.isSelected
                        ? Colors.white.withValues(alpha: 0.2)
                        : theme.colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    widget.count.toString(),
                    style: theme.textTheme.labelSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: widget.isSelected
                          ? Colors.white
                          : theme.colorScheme.onSurfaceVariant,
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
}
