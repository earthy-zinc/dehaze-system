import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../widgets/dehaze_image.dart';
import 'widgets/compare_empty_state.dart';
import 'widgets/comparison_scaffold.dart';

/// 重叠对比页面
///
/// 透明度滑块 + 预设按钮
class OverlayPage extends ConsumerStatefulWidget {
  const OverlayPage({super.key});

  @override
  ConsumerState<OverlayPage> createState() => _OverlayPageState();
}

class _OverlayPageState extends ConsumerState<OverlayPage> {
  double _opacity = 0.5;

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);
    final originalUrl = state.selectedImage?.fileUrl;
    final originalBytes = state.selectedImage?.bytes;
    final resultUrl = state.predictionResult?.resultUrl;

    if (originalUrl == null || resultUrl == null) {
      return ComparisonScaffold(
        icon: Icons.layers_outlined,
        title: '重叠对比',
        body: CompareEmptyState(onAction: () => context.go(AppRouterConfig.processing)),
        currentRoute: AppRouterConfig.overlay,
      );
    }

    return ComparisonScaffold(
      icon: Icons.layers_outlined,
      title: '重叠对比',
      currentRoute: AppRouterConfig.overlay,
      body: _buildImageStack(originalUrl, resultUrl, originalBytes),
      controls: _buildControls(theme),
    );
  }

  Widget _buildImageStack(
    String originalUrl,
    String resultUrl,
    Uint8List? originalBytes,
  ) =>
      Stack(
        children: [
          // 底层：结果图
          Positioned.fill(
            child: DehazeImage(url: resultUrl, fit: BoxFit.contain),
          ),
          // 上层：原图，带透明度
          Positioned.fill(
            child: Opacity(
              opacity: _opacity,
              child: DehazeImage(
                bytes: originalBytes,
                url: originalUrl,
                fit: BoxFit.contain,
              ),
            ),
          ),
          // 标签
          Positioned(
            top: 12,
            right: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '透明度: ${(_opacity * 100).toInt()}%',
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ],
      );

  Widget _buildControls(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Slider(
              value: _opacity,
              min: 0,
              max: 1,
              divisions: 100,
              label: '${(_opacity * 100).toInt()}%',
              onChanged: (v) => setState(() => _opacity = v),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _PresetButton(label: '仅原图', onTap: () => setState(() => _opacity = 1)),
                _PresetButton(label: '半透明', onTap: () => setState(() => _opacity = 0.5)),
                _PresetButton(label: '仅结果', onTap: () => setState(() => _opacity = 0)),
              ],
            ),
          ],
        ),
      );
}

class _PresetButton extends StatelessWidget {
  const _PresetButton({required this.label, required this.onTap});
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => OutlinedButton(onPressed: onTap, child: Text(label));
}
