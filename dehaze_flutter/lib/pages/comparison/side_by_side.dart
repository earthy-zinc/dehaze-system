import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 并排对比页面
///
/// 触控滑动分割线对比原图/结果图
class SideBySidePage extends ConsumerWidget {
  const SideBySidePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);

    final originalUrl = state.selectedImage?.fileUrl;
    final resultUrl = state.predictionResult?.resultUrl;

    if (originalUrl == null || resultUrl == null) {
      return _buildNoData(context, theme);
    }

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme, context),
          Expanded(
            child: _BeforeAfterSlider(
              beforeUrl: originalUrl,
              afterUrl: resultUrl,
            ),
          ),
          _buildBottomNav(context),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme, BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(color: theme.dividerColor),
          ),
        ),
        child: Row(
          children: [
            Icon(Icons.view_column_outlined, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('并排对比', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const Spacer(),
            Text('滑动分割线对比效果',
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('重叠对比'), onPressed: () => context.go(AppRouterConfig.overlay)),
            ActionChip(label: const Text('放大镜'), onPressed: () => context.go(AppRouterConfig.magnifier)),
            ActionChip(label: const Text('滤镜调节'), onPressed: () => context.go(AppRouterConfig.filter)),
            ActionChip(label: const Text('指标评估'), onPressed: () => context.go(AppRouterConfig.metrics)),
          ],
        ),
      );

  Widget _buildNoData(BuildContext context, ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.warning_amber, size: 64, color: theme.colorScheme.error),
            const SizedBox(height: 16),
            Text('请先完成去雾处理', style: theme.textTheme.titleMedium),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => context.go(AppRouterConfig.processing),
              child: const Text('去处理'),
            ),
          ],
        ),
      );
}

/// 滑动分割线对比组件
class _BeforeAfterSlider extends StatefulWidget {
  const _BeforeAfterSlider({required this.beforeUrl, required this.afterUrl});

  final String beforeUrl;
  final String afterUrl;

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
          Positioned.fill(child: _buildImage(widget.afterUrl)),

          // 前图（原图），用 ClipRect 裁剪
          ClipRect(
            clipper: _LeftClipper(_position),
            child: Positioned.fill(child: _buildImage(widget.beforeUrl)),
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

  Widget _buildImage(String url) {
    if (url.startsWith('http')) {
      return Image.network(url, fit: BoxFit.cover);
    }
    return Image.file(File(url), fit: BoxFit.cover);
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
