import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../widgets/dehaze_image.dart';
import 'widgets/compare_empty_state.dart';
import 'widgets/comparison_scaffold.dart';

/// 并排对比页面
///
/// 触控滑动分割线对比原图/结果图
class SideBySidePage extends ConsumerWidget {
  const SideBySidePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(processingProvider);

    final originalUrl = state.selectedImage?.fileUrl;
    final originalBytes = state.selectedImage?.bytes;
    final resultUrl = state.predictionResult?.resultUrl;

    if (originalUrl == null || resultUrl == null) {
      return ComparisonScaffold(
        icon: Icons.view_column_outlined,
        title: '并排对比',
        subtitle: '滑动分割线对比效果',
        body: CompareEmptyState(onAction: () => context.go(AppRouterConfig.processing)),
        currentRoute: AppRouterConfig.sideBySide,
      );
    }

    return ComparisonScaffold(
      icon: Icons.view_column_outlined,
      title: '并排对比',
      subtitle: '滑动分割线对比效果',
      currentRoute: AppRouterConfig.sideBySide,
      body: _BeforeAfterSlider(
        beforeUrl: originalUrl,
        beforeBytes: originalBytes,
        afterUrl: resultUrl,
      ),
    );
  }
}

/// 滑动分割线对比组件
class _BeforeAfterSlider extends StatefulWidget {
  const _BeforeAfterSlider({
    required this.beforeUrl,
    required this.afterUrl,
    this.beforeBytes,
  });

  final String beforeUrl;
  final String afterUrl;
  final Uint8List? beforeBytes;

  @override
  State<_BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

class _BeforeAfterSliderState extends State<_BeforeAfterSlider> {
  double _position = 0.5;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onHorizontalDragUpdate: (details) {
        final box = context.findRenderObject() as RenderBox;
        setState(() {
          _position = (details.localPosition.dx / box.size.width).clamp(0.0, 1.0);
        });
      },
      child: Stack(
        children: [
          // 后图（结果图）
          Positioned.fill(
            child: DehazeImage(url: widget.afterUrl, fit: BoxFit.cover),
          ),

          // 前图（原图），用 ClipRect 裁剪
          ClipRect(
            clipper: _LeftClipper(_position),
            child: Positioned.fill(
              child: DehazeImage(
                bytes: widget.beforeBytes,
                url: widget.beforeUrl,
                fit: BoxFit.cover,
              ),
            ),
          ),

          // 分割线
          Positioned(
            left: 0,
            top: 0,
            bottom: 0,
            width: MediaQuery.of(context).size.width * _position,
            child: Align(
              alignment: Alignment.centerRight,
              child: Container(
                width: 3,
                color: Colors.white,
                child: Center(
                  child: Container(
                    width: 32,
                    height: 32,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.2), blurRadius: 8)],
                    ),
                    child: const Icon(Icons.drag_indicator, size: 18),
                  ),
                ),
              ),
            ),
          ),

          // 标签
          Positioned(
            top: 12,
            left: 12,
            child: _Label('原图'),
          ),
          Positioned(
            top: 12,
            right: 12,
            child: _Label('去雾结果'),
          ),
        ],
      ),
    );
  }
}

class _LeftClipper extends CustomClipper<Rect> {
  _LeftClipper(this.position);
  final double position;

  @override
  Rect getClip(Size size) => Rect.fromLTWH(0, 0, size.width * position, size.height);

  @override
  bool shouldReclip(_LeftClipper oldClipper) => position != oldClipper.position;
}

class _Label extends StatelessWidget {
  const _Label(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.black.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w500)),
      );
}
