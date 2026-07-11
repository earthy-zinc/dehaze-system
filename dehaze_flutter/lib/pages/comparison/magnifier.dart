import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 放大镜对比页面
///
/// 触控移动放大镜，可切换查看原图/结果图
class MagnifierPage extends ConsumerStatefulWidget {
  const MagnifierPage({super.key});

  @override
  ConsumerState<MagnifierPage> createState() => _MagnifierPageState();
}

class _MagnifierPageState extends ConsumerState<MagnifierPage> {
  Offset _position = Offset.zero;
  double _lensSize = 120;
  bool _showOriginal = true;
  bool _initialized = false;

  @override
  Widget build(BuildContext context) {
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
            child: LayoutBuilder(
              builder: (context, constraints) {
                if (!_initialized) {
                  _position = Offset(constraints.maxWidth / 2, constraints.maxHeight / 2);
                  _initialized = true;
                }
                return GestureDetector(
                  onPanUpdate: (details) {
                    setState(() {
                      _position = Offset(
                        (_position.dx + details.delta.dx).clamp(0, constraints.maxWidth),
                        (_position.dy + details.delta.dy).clamp(0, constraints.maxHeight),
                      );
                    });
                  },
                  child: Stack(
                    children: [
                      // 底层图片
                      Positioned.fill(child: _buildImage(resultUrl)),
                      // 放大镜
                      Positioned(
                        left: _position.dx - _lensSize / 2,
                        top: _position.dy - _lensSize / 2,
                        child: ClipOval(
                          child: Container(
                            width: _lensSize,
                            height: _lensSize,
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.white, width: 3),
                              boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.3), blurRadius: 8)],
                            ),
                            child: Stack(
                              children: [
                                // 放大的图片
                                Positioned.fill(
                                  child: FittedBox(
                                    fit: BoxFit.none,
                                    alignment: Alignment(
                                      -1 + 2 * (_position.dx / constraints.maxWidth),
                                      -1 + 2 * (_position.dy / constraints.maxHeight),
                                    ),
                                    child: SizedBox(
                                      width: constraints.maxWidth * 2,
                                      height: constraints.maxHeight * 2,
                                      child: _buildImage(_showOriginal ? originalUrl : resultUrl),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
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
            Icon(Icons.search_outlined, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('放大镜对比', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          ],
        ),
      );

  Widget _buildControls(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SegmentedButton<bool>(
                  segments: const [
                    ButtonSegment(value: true, label: Text('看原图')),
                    ButtonSegment(value: false, label: Text('看结果')),
                  ],
                  selected: {_showOriginal},
                  onSelectionChanged: (v) => setState(() => _showOriginal = v.first),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Text('镜片大小'),
                Expanded(
                  child: Slider(
                    value: _lensSize,
                    min: 60,
                    max: 240,
                    divisions: 18,
                    label: '${_lensSize.toInt()}',
                    onChanged: (v) => setState(() => _lensSize = v),
                  ),
                ),
                Text('${_lensSize.toInt()}px'),
              ],
            ),
          ],
        ),
      );

  Widget _buildImage(String url) {
    if (url.startsWith('http')) return Image.network(url, fit: BoxFit.cover);
    return Image.file(File(url), fit: BoxFit.cover);
  }

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('并排对比'), onPressed: () => context.go(AppRouterConfig.sideBySide)),
            ActionChip(label: const Text('重叠对比'), onPressed: () => context.go(AppRouterConfig.overlay)),
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
