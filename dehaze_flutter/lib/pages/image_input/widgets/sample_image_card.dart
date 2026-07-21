import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';

import '../models/image_input_model.dart';

/// 样例图片卡片
///
/// 显示样例图片缩略图、名称和难度标签
class SampleImageCard extends StatefulWidget {
  const SampleImageCard({
    required this.sample,
    super.key,
    this.onTap,
  });

  final SampleImageModel sample;
  final VoidCallback? onTap;

  @override
  State<SampleImageCard> createState() => _SampleImageCardState();
}

class _SampleImageCardState extends State<SampleImageCard> {
  bool _isHovered = false;
  bool _isPressed = false;

  Color get _difficultyColor {
    switch (widget.sample.difficulty) {
      case DifficultyLevel.easy:
        return const Color(0xFF10B981); // emerald-500
      case DifficultyLevel.medium:
        return const Color(0xFFF59E0B); // amber-500
      case DifficultyLevel.hard:
        return const Color(0xFFEF4444); // red-500
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
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(12),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: _isHovered ? 0.12 : 0.06),
                blurRadius: _isHovered ? 16 : 8,
                offset: Offset(0, _isHovered ? 8 : 4),
              ),
            ],
          ),
          clipBehavior: Clip.antiAlias,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 图片
              Expanded(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CachedNetworkImage(
                      imageUrl: widget.sample.url,
                      fit: BoxFit.cover,
                      placeholder: (context, url) => Container(
                        color: theme.colorScheme.surfaceContainerHighest,
                        child: const Center(
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ),
                      errorWidget: (context, url, error) => Container(
                        color: theme.colorScheme.surfaceContainerHighest,
                        child: Icon(
                          Icons.broken_image_outlined,
                          size: 32,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                    // 悬停遮罩
                    AnimatedOpacity(
                      duration: const Duration(milliseconds: 200),
                      opacity: _isHovered ? 1.0 : 0.0,
                      child: Container(
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              Colors.transparent,
                              Colors.black.withValues(alpha: 0.5),
                            ],
                          ),
                        ),
                        child: const Center(
                          child: Icon(
                            Icons.touch_app_outlined,
                            color: Colors.white,
                            size: 32,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // 信息区域
              Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 名称
                    Text(
                      widget.sample.name,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: theme.colorScheme.onSurface,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 8),
                    // 标签行
                    Row(
                      children: [
                        // 难度标签
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: _difficultyColor.withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            widget.sample.difficulty.displayName,
                            style: theme.textTheme.labelSmall?.copyWith(
                              fontWeight: FontWeight.w600,
                              color: _difficultyColor,
                            ),
                          ),
                        ),
                        const Spacer(),
                        // 箭头
                        Icon(
                          Icons.arrow_forward,
                          size: 16,
                          color: theme.colorScheme.primary,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
