import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 滤镜调节页面
///
/// 亮度/对比度/饱和度 3个滤镜滑块 + 6个预设方案
class FilterPage extends ConsumerStatefulWidget {
  const FilterPage({super.key});

  @override
  ConsumerState<FilterPage> createState() => _FilterPageState();
}

class _FilterPageState extends ConsumerState<FilterPage> {
  double _brightness = 0;
  double _contrast = 0;
  double _saturation = 0;

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);
    final resultUrl = state.predictionResult?.resultUrl;

    if (resultUrl == null) {
      return _buildNoData(context, theme);
    }

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme),
          Expanded(child: _buildImageWithFilter(resultUrl)),
          _buildControls(theme),
          _buildBottomNav(context),
        ],
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(bottom: BorderSide(color: theme.dividerColor)),
        ),
        child: Row(
          children: [
            Icon(Icons.tune_outlined, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('滤镜调节', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          ],
        ),
      );

  Widget _buildImageWithFilter(String url) => Center(
        child: ColorFiltered(
          colorFilter: ui.ColorFilter.matrix(_buildColorMatrix()),
          child: url.startsWith('http')
              ? Image.network(url, fit: BoxFit.contain)
              : Image.file(File(url), fit: BoxFit.contain),
        ),
      );

  List<double> _buildColorMatrix() {
    final b = 1 + _brightness;
    final c = 1 + _contrast;
    final s = 1 + _saturation;
    return <double>[
      c * b, 0, 0, 0, _brightness * 255 * (1 - c),
      0, c * b, 0, 0, _brightness * 255 * (1 - c),
      0, 0, c * b, 0, _brightness * 255 * (1 - c),
      0, 0, 0, s, 0,
    ];
  }

  Widget _buildControls(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            _buildSlider('亮度', _brightness, (v) => setState(() => _brightness = v)),
            _buildSlider('对比度', _contrast, (v) => setState(() => _contrast = v)),
            _buildSlider('饱和度', _saturation, (v) => setState(() => _saturation = v)),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: [
                _PresetChip('原图', () => _applyPreset(0, 0, 0)),
                _PresetChip('提亮', () => _applyPreset(0.15, 0, 0)),
                _PresetChip('增强', () => _applyPreset(0.1, 0.15, 0.2)),
                _PresetChip('柔和', () => _applyPreset(-0.05, -0.1, -0.15)),
                _PresetChip('鲜艳', () => _applyPreset(0, 0.1, 0.4)),
                _PresetChip('黑白', () => _applyPreset(0, 0, -1)),
              ],
            ),
          ],
        ),
      );

  void _applyPreset(double b, double c, double s) {
    setState(() {
      _brightness = b;
      _contrast = c;
      _saturation = s;
    });
  }

  Widget _buildSlider(String label, double value, ValueChanged<double> onChanged) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            SizedBox(width: 60, child: Text(label)),
            Expanded(
              child: Slider(
                value: value,
                min: -1,
                max: 1,
                divisions: 20,
                label: value.toStringAsFixed(1),
                onChanged: onChanged,
              ),
            ),
            SizedBox(
              width: 40,
              child: Text(value.toStringAsFixed(1), textAlign: TextAlign.right),
            ),
          ],
        ),
      );

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('并排对比'), onPressed: () => context.go(AppRouterConfig.sideBySide)),
            ActionChip(label: const Text('重叠对比'), onPressed: () => context.go(AppRouterConfig.overlay)),
            ActionChip(label: const Text('放大镜'), onPressed: () => context.go(AppRouterConfig.magnifier)),
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

class _PresetChip extends StatelessWidget {
  const _PresetChip(this.label, this.onTap);
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => ActionChip(label: Text(label), onPressed: onTap);
}
