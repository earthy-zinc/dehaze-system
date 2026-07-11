import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

/// 类型筛选标签组件
///
/// 支持横向滚动，适配移动端
class TypeFilterTabs extends StatelessWidget {
  const TypeFilterTabs({
    required this.selectedType,
    required this.onTypeChanged,
    this.totalCount,
    this.hazyCount,
    this.clearCount,
    this.dehazedCount,
    super.key,
  });

  final ImageType? selectedType;
  final int? totalCount;
  final int? hazyCount;
  final int? clearCount;
  final int? dehazedCount;
  final void Function(ImageType?) onTypeChanged;

  @override
  Widget build(BuildContext context) {
    final filterOptions = [
      _FilterOption(null, '全部', totalCount ?? 0),
      _FilterOption(ImageType.hazy, '有雾', hazyCount ?? 0),
      _FilterOption(ImageType.clear, '清晰', clearCount ?? 0),
      _FilterOption(ImageType.dehazed, '去雾结果', dehazedCount ?? 0),
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

    const tealColor = Color(0xFF14B8A6);
    const cyanColor = Color(0xFF06B6D4);

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
                color: widget.isSelected ? null : theme.colorScheme.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: widget.isSelected
                      ? tealColor
                      : _isHovered
                          ? tealColor
                          : const Color(0xFFE5E7EB),
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
                              : const Color(0xFF6B7280),
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
