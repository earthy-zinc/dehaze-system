import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/dehaze_params.dart';
import '../../providers/processing_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import '../../widgets/dehaze_image.dart';

/// 去雾处理页面
class ProcessingPage extends ConsumerStatefulWidget {
  const ProcessingPage({super.key});

  @override
  ConsumerState<ProcessingPage> createState() => _ProcessingPageState();
}

class _ProcessingPageState extends ConsumerState<ProcessingPage> {
  // 处理参数（与文档定义一致）
  double _strength = 50; // 去雾强度 0-100
  double _saturation = 100; // 饱和度 0-200
  double _contrast = 100; // 对比度 0-200
  double _sharpen = 30; // 锐化 0-100

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(processingProvider);
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1000,
        child: CustomScrollView(
          slivers: [
            // 页面头部
            SliverToBoxAdapter(child: _buildHeader(theme)),

            // 内容
            SliverPadding(
              padding: ResponsiveUtils.getResponsivePadding(context),
              sliver: SliverToBoxAdapter(
                child: _buildBody(theme, state),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        padding: ResponsiveUtils.getResponsivePadding(context),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(
            bottom: BorderSide(color: theme.dividerColor, width: 1),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                IconButton(
                  onPressed: () => context.go(AppRouterConfig.imageInput),
                  icon: const Icon(Icons.arrow_back),
                  tooltip: '返回',
                ),
                Icon(Icons.auto_fix_high, color: AppTheme.brandBlue, size: 24),
                const SizedBox(width: 8),
                Text(
                  '去雾处理',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '调节参数并开始处理',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );

  Widget _buildBody(ThemeData theme, ProcessingState state) {
    // 检查是否有选中的图片和算法
    if (state.selectedImage == null || state.selectedAlgorithm == null) {
      return _buildMissingSelection(theme);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 图片预览
        _buildImagePreview(theme, state),

        const SizedBox(height: 24),

        // 算法信息
        _buildAlgorithmInfo(theme, state),

        const SizedBox(height: 24),

        // 参数调节
        _buildParamsSection(theme),

        const SizedBox(height: 24),

        // 处理按钮 / 进度 / 结果
        _buildProcessingSection(theme, state),

        const SizedBox(height: 32),
      ],
    );
  }

  Widget _buildMissingSelection(ThemeData theme) => Center(
        child: Padding(
          padding: const EdgeInsets.all(48),
          child: Column(
            children: [
              Icon(Icons.warning_amber_rounded,
                  size: 64, color: theme.colorScheme.error),
              const SizedBox(height: 16),
              Text('请先选择图片和算法',
                  style: theme.textTheme.titleMedium),
              const SizedBox(height: 16),
              Wrap(
                spacing: 12,
                children: [
                  OutlinedButton.icon(
                    onPressed: () =>
                        context.go(AppRouterConfig.imageInput),
                    icon: const Icon(Icons.image),
                    label: const Text('选择图片'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () =>
                        context.go(AppRouterConfig.algorithmSelect),
                    icon: const Icon(Icons.psychology),
                    label: const Text('选择算法'),
                  ),
                ],
              ),
            ],
          ),
        ),
      );

  Widget _buildImagePreview(ThemeData theme, ProcessingState state) {
    final image = state.selectedImage!;
    final resultUrl = state.predictionResult?.resultUrl;

    return Container(
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.colorScheme.outline),
      ),
      child: resultUrl != null
          ? _buildBeforeAfterView(theme, image, resultUrl)
          : _buildSingleImage(theme, image),
    );
  }

  Widget _buildSingleImage(ThemeData theme, SelectedImage image) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
      child: AspectRatio(
        aspectRatio: 16 / 10,
        child: _imageWidget(image.fileUrl, bytes: image.bytes),
      ),
    );
  }

  Widget _buildBeforeAfterView(
    ThemeData theme,
    SelectedImage image,
    String resultUrl,
  ) {
    final isWide = ResponsiveUtils.isWideScreen(context);

    if (isWide) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Expanded(
                child: _labeledImage('原图', image.fileUrl, bytes: image.bytes)),
            const SizedBox(width: 12),
            Expanded(child: _labeledImage('去雾结果', resultUrl)),
          ],
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _labeledImage('原图', image.fileUrl, bytes: image.bytes),
          const SizedBox(height: 12),
          _labeledImage('去雾结果', resultUrl),
        ],
      ),
    );
  }

  Widget _labeledImage(String label, String url, {Uint8List? bytes}) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
            child: AspectRatio(
              aspectRatio: 16 / 10,
              child: _imageWidget(url, bytes: bytes),
            ),
          ),
        ],
      );

  Widget _imageWidget(String url, {Uint8List? bytes}) {
    // 跨平台渲染：字节流优先（原图），网络地址次之（结果图）
    return DehazeImage(
      bytes: bytes,
      url: url,
      fit: BoxFit.cover,
    );
  }

  Widget _buildAlgorithmInfo(ThemeData theme, ProcessingState state) =>
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          border: Border.all(color: theme.colorScheme.outline),
        ),
        child: Row(
          children: [
            Icon(Icons.psychology_outlined,
                color: AppTheme.brandBlue, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    state.selectedAlgorithm!.name,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    state.selectedAlgorithm!.type,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ],
              ),
            ),
            TextButton(
              onPressed: () =>
                  context.go(AppRouterConfig.algorithmSelect),
              child: const Text('更换'),
            ),
          ],
        ),
      );

  Widget _buildParamsSection(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          border: Border.all(color: theme.colorScheme.outline),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('参数调节',
                style: theme.textTheme.titleMedium
                    ?.copyWith(fontWeight: FontWeight.w600)),
            const SizedBox(height: 16),
            _buildSlider('去雾强度', _strength, 0, 100,
                (v) => setState(() => _strength = v)),
            const SizedBox(height: 12),
            _buildSlider('饱和度', _saturation, 0, 200,
                (v) => setState(() => _saturation = v)),
            const SizedBox(height: 12),
            _buildSlider('对比度', _contrast, 0, 200,
                (v) => setState(() => _contrast = v)),
            const SizedBox(height: 12),
            _buildSlider('锐化', _sharpen, 0, 100,
                (v) => setState(() => _sharpen = v)),
          ],
        ),
      );

  Widget _buildSlider(
    String label,
    double value,
    double min,
    double max,
    ValueChanged<double> onChanged,
  ) =>
      Row(
        children: [
          SizedBox(
            width: 64,
            child: Text(label),
          ),
          Expanded(
            child: Slider(
              value: value,
              min: min,
              max: max,
              divisions: ((max - min) / 5).round(),
              label: value.toStringAsFixed(0),
              onChanged: onChanged,
            ),
          ),
          SizedBox(
            width: 44,
            child: Text(
              value.toStringAsFixed(0),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      );

  Widget _buildProcessingSection(ThemeData theme, ProcessingState state) {
    switch (state.status) {
      case ProcessingStatus.idle:
        return FilledButton.icon(
          onPressed: state.canProcess
              ? () => _startProcessing(ref)
              : null,
          icon: const Icon(Icons.play_arrow),
          label: const Text('开始去雾处理'),
          style: FilledButton.styleFrom(
            minimumSize: const Size(double.infinity, 52),
          ),
        );

      case ProcessingStatus.processing:
        return _buildProcessingIndicator(state);

      case ProcessingStatus.success:
        return Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.techGreen.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(AppTheme.radiusL),
                border: Border.all(color: AppTheme.techGreen.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  Icon(Icons.check_circle, color: AppTheme.techGreen, size: 24),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('处理完成!',
                            style: TextStyle(
                              fontWeight: FontWeight.w600,
                              color: AppTheme.techGreen,
                            )),
                        if (state.predictionResult?.time != null)
                          Text(
                            '耗时: ${(state.predictionResult!.time! / 1000).toStringAsFixed(1)}秒',
                            style: theme.textTheme.bodySmall,
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => context.go(AppRouterConfig.sideBySide),
                    icon: const Icon(Icons.compare),
                    label: const Text('效果对比'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () {
                      ref.read(processingProvider.notifier).reset();
                      context.go(AppRouterConfig.imageInput);
                    },
                    icon: const Icon(Icons.refresh),
                    label: const Text('继续处理'),
                  ),
                ),
              ],
            ),
          ],
        );

      case ProcessingStatus.error:
        return Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: theme.colorScheme.error.withValues(alpha: 0.05),
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            border: Border.all(color: theme.colorScheme.error.withValues(alpha: 0.3)),
          ),
          child: Column(
            children: [
              Icon(Icons.error_outline, color: theme.colorScheme.error),
              const SizedBox(height: 8),
              Text(state.errorMessage ?? '处理失败'),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: () {
                  ref.read(processingProvider.notifier).clearError();
                  _startProcessing(ref);
                },
                icon: const Icon(Icons.refresh),
                label: const Text('重试'),
              ),
            ],
          ),
        );
    }
  }

  /// 构建处理中指示器：进度环 + 真实已用时间（不显示假百分比）
  ///
  /// 已用时间由独立的 [_TimerText] 自行驱动定时刷新，
  /// 避免每 100ms 触发整棵 Widget 树重建。
  Widget _buildProcessingIndicator(ProcessingState state) {
    final startTime = state.processingStartTime;

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AppTheme.brandBlue.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Column(
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          const Text('正在处理...'),
          if (startTime != null) ...[
            const SizedBox(height: 8),
            _TimerText(start: startTime),
          ],
        ],
      ),
    );
  }

  void _startProcessing(WidgetRef ref) {
    final params = DehazeParams(
      strength: _strength.round(),
      saturation: _saturation.round(),
      contrast: _contrast.round(),
      sharpen: _sharpen.round(),
    );
    ref.read(processingProvider.notifier).process(params: params);
  }
}

/// 处理已用时间文本
///
/// 独立维护定时器与已用秒数，仅自身重建，
/// 避免父页面定时刷新导致整棵子树重建。
class _TimerText extends StatefulWidget {
  const _TimerText({required this.start});

  final DateTime start;

  @override
  State<_TimerText> createState() => _TimerTextState();
}

class _TimerTextState extends State<_TimerText> {
  Timer? _timer;
  int _elapsedSeconds = 0;

  @override
  void initState() {
    super.initState();
    _elapsedSeconds = DateTime.now().difference(widget.start).inSeconds;
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _elapsedSeconds = DateTime.now().difference(widget.start).inSeconds;
        });
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Text(
      '已用 ${_elapsedSeconds}s',
      style: theme.textTheme.bodySmall?.copyWith(
        color: theme.colorScheme.onSurfaceVariant,
      ),
    );
  }
}
