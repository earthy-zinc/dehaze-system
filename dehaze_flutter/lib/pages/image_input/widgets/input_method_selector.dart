import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../utils/responsive_utils.dart';
import '../models/image_input_model.dart';
import '../providers/image_input_provider.dart';

/// 输入方式选择器
///
/// 4个输入方式按钮的网格布局
/// 支持响应式：移动端 2x2，桌面端 1x4
class InputMethodSelector extends ConsumerWidget {
  const InputMethodSelector({super.key, this.onMethodChanged});

  final ValueChanged<InputMethod>? onMethodChanged;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final currentMethod = ref.watch(inputMethodProvider);
    final isWide = ResponsiveUtils.isWideScreen(context);

    return GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: isWide ? 4 : 2,
      mainAxisSpacing: 12,
      crossAxisSpacing: 12,
      childAspectRatio: isWide ? 1.5 : 1.2,
      children: InputMethod.values.map((method) {
        return _InputMethodButton(
          method: method,
          isSelected: currentMethod == method,
          onTap: () {
            ref.read(inputMethodProvider.notifier).state = method;
            onMethodChanged?.call(method);
          },
        );
      }).toList(),
    );
  }
}

/// 输入方式按钮
class _InputMethodButton extends StatefulWidget {
  const _InputMethodButton({
    required this.method,
    required this.isSelected,
    required this.onTap,
  });

  final InputMethod method;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  State<_InputMethodButton> createState() => _InputMethodButtonState();
}

class _InputMethodButtonState extends State<_InputMethodButton> {
  bool _isHovered = false;
  bool _isPressed = false;

  IconData get _icon {
    switch (widget.method) {
      case InputMethod.upload:
        return Icons.cloud_upload_outlined;
      case InputMethod.camera:
        return Icons.camera_alt_outlined;
      case InputMethod.sample:
        return Icons.collections_outlined;
      case InputMethod.history:
        return Icons.history_outlined;
    }
  }

  Color get _iconColor {
    switch (widget.method) {
      case InputMethod.upload:
        return const Color(0xFF3B82F6); // blue-500
      case InputMethod.camera:
        return const Color(0xFF10B981); // emerald-500
      case InputMethod.sample:
        return const Color(0xFF8B5CF6); // violet-500
      case InputMethod.history:
        return const Color(0xFFF59E0B); // amber-500
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTapDown: (_) => setState(() => _isPressed = true),
        onTapUp: (_) => setState(() => _isPressed = false),
        onTapCancel: () => setState(() => _isPressed = false),
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          transform: Matrix4.diagonal3Values(
              _isPressed ? 0.95 : 1.0, _isPressed ? 0.95 : 1.0, 1.0),
          transformAlignment: Alignment.center,
          decoration: BoxDecoration(
            color: widget.isSelected
                ? _iconColor.withValues(alpha: 0.1)
                : theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: widget.isSelected
                  ? _iconColor
                  : _isHovered
                      ? _iconColor.withValues(alpha: 0.5)
                      : theme.dividerColor,
              width: widget.isSelected ? 2 : 1,
            ),
            boxShadow: _isHovered || widget.isSelected
                ? [
                    BoxShadow(
                      color: _iconColor.withValues(alpha: 0.2),
                      blurRadius: 8,
                      offset: const Offset(0, 4),
                    ),
                  ]
                : null,
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                _icon,
                size: 32,
                color: widget.isSelected ? _iconColor : theme.colorScheme.onSurfaceVariant,
              ),
              const SizedBox(height: 8),
              Text(
                widget.method.displayName,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: widget.isSelected ? FontWeight.w600 : FontWeight.w500,
                  color: widget.isSelected ? _iconColor : theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                widget.method.subtitle,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
