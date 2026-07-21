import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../widgets/dehaze_image.dart';

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
      return _buildNoData(context, theme);
    }

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme, context),
          Expanded(
              child: _buildImageStack(originalUrl, resultUrl, originalBytes)),
          _buildControls(theme),
          _buildBottomNav(context),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme, BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(bottom: BorderSide(color: theme.dividerColor)),
        ),
        child: Row(
          children: [
            Icon(Icons.layers_outlined, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('重叠对比', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          ],
        ),
      );

  Widget _buildImageStack(
    String originalUrl,
    String resultUrl,
    Uint8List? originalBytes,
  ) =>
      Stack(
        children: [
          // 底层：结果图
          Positioned.fill(child: _buildImage(resultUrl)),
          // 上层：原图，带透明度
          Positioned.fill(
            child: Opacity(
              opacity: _opacity,
              child: _buildImage(originalUrl, bytes: originalBytes),
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

  Widget _buildImage(String url, {Uint8List? bytes}) {
    return DehazeImage(
      bytes: bytes,
      url: url,
      fit: BoxFit.contain,
    );
  }

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('并排对比'), onPressed: () => context.go(AppRouterConfig.sideBySide)),
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

class _PresetButton extends StatelessWidget {
  const _PresetButton({required this.label, required this.onTap});
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => OutlinedButton(onPressed: onTap, child: Text(label));
}
