import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../widgets/dehaze_image.dart';
import 'widgets/compare_empty_state.dart';
import 'widgets/comparison_scaffold.dart';

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
    final originalBytes = state.selectedImage?.bytes;
    final resultUrl = state.predictionResult?.resultUrl;

    if (originalUrl == null || resultUrl == null) {
      return ComparisonScaffold(
        icon: Icons.search_outlined,
        title: '放大镜对比',
        body: CompareEmptyState(onAction: () => context.go(AppRouterConfig.processing)),
        currentRoute: AppRouterConfig.magnifier,
      );
    }

    return ComparisonScaffold(
      icon: Icons.search_outlined,
      title: '放大镜对比',
      currentRoute: AppRouterConfig.magnifier,
      body: _buildMagnifierBody(originalUrl, originalBytes, resultUrl),
      controls: _buildControls(theme),
    );
  }

  Widget _buildMagnifierBody(
    String originalUrl,
    Uint8List? originalBytes,
    String resultUrl,
  ) {
    return LayoutBuilder(
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
              Positioned.fill(
                child: DehazeImage(url: resultUrl, fit: BoxFit.cover),
              ),
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
                                      child: DehazeImage(
                                        bytes: _showOriginal ? originalBytes : null,
                                        url: _showOriginal ? originalUrl : resultUrl,
                                        fit: BoxFit.cover,
                                      ),
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
    );
  }

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
}
