import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

/// 类型筛选标签组件
///
/// 与设计稿 dataset.css 的 type-filter-btn 样式对应
/// 支持横向滚动，适配移动端
class TypeFilterTabs extends StatelessWidget {
  const TypeFilterTabs({
    required this.selectedType,
    required this.totalCount,
    required this.foggyCount,
    required this.clearCount,
    required this.annotatedCount,
    required this.onTypeChanged,
    super.key,
  });

  final ImageType? selectedType;
  final int totalCount;
  final int foggyCount;
  final int clearCount;
  final int annotatedCount;
  final void Function(ImageType?) onTypeChanged;

  @override
  Widget build(BuildContext context) {
    final filterOptions = [
      _FilterOption(null, '全部', totalCount),
      _FilterOption(ImageType.foggy, '有雾', foggyCount),
      _FilterOption(ImageType.clear, '无雾', clearCount),
      _FilterOption(ImageType.annotated, '标注', annotatedCount),
    ];

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: AppTheme.spacingM,
        vertical: AppTheme.spacingS,
      ),
      child: SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        child: Row(
          children: filterOptions.map((option) {
            final isSelected = selectedType == option.type;

            return Padding(
              padding: EdgeInsets.only(right: AppTheme.spacingS),
              child: _FilterButton(
                label: option.label,
                count: option.count,
                isSelected: isSelected,
                onTap: () => onTypeChanged(option.type),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }
}

/// 筛选按钮组件
///
/// 与设计稿 type-filter-btn 样式完全对应
class _FilterButton extends StatefulWidget {
  const _FilterButton({
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
  State<_FilterButton> createState() => _FilterButtonState();
}

class _FilterButtonState extends State<_FilterButton> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // 设计稿颜色
    const tealColor = Color(0xFF14B8A6); // Tailwind teal-500
    const cyanColor = Color(0xFF06B6D4); // Tailwind cyan-500

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: widget.onTap,
            borderRadius: BorderRadius.circular(20),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                gradient: widget.isSelected
                    ? const LinearGradient(
                        colors: [tealColor, cyanColor],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      )
                    : null,
                color: widget.isSelected ? null : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: widget.isSelected
                      ? tealColor
                      : _isHovered
                          ? tealColor
                          : const Color(0xFFE5E7EB), // gray-200
                  width: 2,
                ),
                boxShadow: widget.isSelected
                    ? [
                        BoxShadow(
                          color: tealColor.withValues(alpha: 0.3),
                          blurRadius: 12,
                          offset: const Offset(0, 4),
                        ),
                      ]
                    : null,
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    widget.label,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: widget.isSelected
                          ? Colors.white
                          : _isHovered
                              ? tealColor
                              : const Color(0xFF6B7280), // gray-500
                    ),
                  ),
                  if (widget.count > 0) ...[
                    const SizedBox(width: 4),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: widget.isSelected
                            ? Colors.white.withValues(alpha: 0.2)
                            : theme.colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '${widget.count}',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
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
        ),
      ),
    );
  }
}

class _FilterOption {
  const _FilterOption(this.type, this.label, this.count);
  final ImageType? type;
  final String label;
  final int count;
}
