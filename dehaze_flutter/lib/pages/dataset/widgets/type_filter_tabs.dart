import 'package:flutter/material.dart';
import '../../../theme/app_theme.dart';
import '../models/dataset_model.dart';

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
    final theme = Theme.of(context);

    final filterOptions = [
      _FilterOption(null, '全部', totalCount),
      _FilterOption(ImageType.foggy, '有雾', foggyCount),
      _FilterOption(ImageType.clear, '无雾', clearCount),
      _FilterOption(ImageType.annotated, '标注', annotatedCount),
    ];

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: EdgeInsets.symmetric(vertical: AppTheme.spacingS),
      child: Row(
        children: filterOptions.map((option) {
          final isSelected = selectedType == option.type;

          return Padding(
            padding: EdgeInsets.only(right: AppTheme.spacingS),
            child: FilterChip(
              selected: isSelected,
              onSelected: (selected) {
                if (selected) {
                  onTypeChanged(option.type);
                }
              },
              label: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(option.label),
                  if (option.count > 0) ...[
                    SizedBox(width: AppTheme.spacingXS),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? Colors.white.withValues(alpha: 0.2)
                            : theme.colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '${option.count}',
                        style: TextStyle(
                          fontSize: 12,
                          color: isSelected
                              ? Colors.white
                              : theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                  ],
                ],
              ),
              backgroundColor: theme.colorScheme.surface,
              selectedColor: theme.colorScheme.primary,
              checkmarkColor: Colors.white,
              side: BorderSide(
                color: isSelected
                    ? theme.colorScheme.primary
                    : theme.colorScheme.outline,
                width: 1,
              ),
            ),
          );
        }).toList(),
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
