import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/constants/storage_constants.dart';
import '../../models/ai_model_model.dart';
import '../../models/dehaze_params.dart';
import '../../providers/processing_provider.dart';
import '../../providers/providers.dart';
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
  // 处理参数（默认值取自 DehazeParams，避免多处硬编码）
  double _strength = DehazeParams.defaultStrength.toDouble();
  double _saturation = DehazeParams.defaultSaturation.toDouble();
  double _contrast = DehazeParams.defaultContrast.toDouble();
  double _sharpen = DehazeParams.defaultSharpen.toDouble();

  @override
  void initState() {
    super.initState();
    // 未手动选择时：历史最近使用的模型（须仍在启用列表内）→ 启用列表首项
    ref.listenManual(imageEditModelsProvider, (_, next) {
      final models = next.valueOrNull;
      if (models == null || models.isEmpty) return;
      if (ref.read(processingProvider).selectedModel != null) return;
      final lastUsed = ref
          .read(sharedPreferencesProvider)
          .getString(StorageConstants.lastImageModelId);
      final preferred = models.where((m) => m.modelId == lastUsed).firstOrNull;
      ref
          .read(processingProvider.notifier)
          .setSelectedModel(preferred ?? models.first);
    });
  }

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
              sliver: SliverToBoxAdapter(child: _buildBody(theme, state)),
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
      border: Border(bottom: BorderSide(color: theme.dividerColor, width: 1)),
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
    // 检查是否有选中的图片
    if (state.selectedImage == null) {
      return _buildMissingSelection(theme);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 图片预览
        _buildImagePreview(theme, state),

        const SizedBox(height: 24),

        // 模型选择（内联，数据源为统一模型注册表）
        _buildModelSelector(theme),

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
          Icon(
            Icons.warning_amber_rounded,
            size: 64,
            color: theme.colorScheme.error,
          ),
          const SizedBox(height: 16),
          Text('请先选择图片', style: theme.textTheme.titleMedium),
          const SizedBox(height: 16),
          OutlinedButton.icon(
            onPressed: () => context.go(AppRouterConfig.imageInput),
            icon: const Icon(Icons.image),
            label: const Text('选择图片'),
          ),
        ],
      ),
    ),
  );

  Widget _buildImagePreview(ThemeData theme, ProcessingState state) {
    final image = state.selectedImage!;
    final resultUrl = state.taskResult?.resultUrl;

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
              child: _labeledImage('原图', image.fileUrl, bytes: image.bytes),
            ),
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
      Text(label, style: const TextStyle(fontWeight: FontWeight.w600)),
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
    return DehazeImage(bytes: bytes, url: url, fit: BoxFit.cover);
  }

  /// 内联模型选择器
  ///
  /// 数据源为 GET /ai/models/enabled?modelType=image_edit，
  /// 选中项直接写入处理流程状态，无需跳转独立选择页。
  Widget _buildModelSelector(ThemeData theme) {
    final modelsAsync = ref.watch(imageEditModelsProvider);
    final selected = ref.watch(processingProvider).selectedModel;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.colorScheme.outline),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.psychology_outlined,
                color: AppTheme.brandBlue,
                size: 20,
              ),
              const SizedBox(width: 12),
              Text(
                '处理模型',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          modelsAsync.when(
            loading: () => const LinearProgressIndicator(),
            error: (error, _) => Text(
              '模型列表加载失败：$error',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
            data: (models) {
              if (models.isEmpty) {
                return Text(
                  '暂无可用模型',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                );
              }
              // 按 modelId 回落选中项，避免列表刷新后实例不等导致下拉断言失败
              final current = models
                  .where((m) => m.modelId == selected?.modelId)
                  .firstOrNull;
              final groups = _groupModelsByTaskTag(models).entries;
              return DropdownButtonFormField<AiModelVO>(
                initialValue: current,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  isDense: true,
                ),
                items: [
                  for (final group in groups) ...[
                    // 组标题禁用：不可点击/聚焦，仅作视觉分区，不影响表单校验
                    DropdownMenuItem<AiModelVO>(
                      enabled: false,
                      child: Text(
                        group.key,
                        style: theme.textTheme.labelLarge?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    for (final model in group.value)
                      DropdownMenuItem<AiModelVO>(
                        value: model,
                        child: Text(model.displayName),
                      ),
                  ],
                ],
                onChanged: (model) {
                  if (model != null) {
                    ref
                        .read(processingProvider.notifier)
                        .setSelectedModel(model);
                  }
                },
              );
            },
          ),
        ],
      ),
    );
  }

  /// 按首个任务标签归属分组：每个模型只出现一次，分组顺序取标签首次出现顺序，
  /// 无标签模型归入「其他」并置末。
  Map<String, List<AiModelVO>> _groupModelsByTaskTag(List<AiModelVO> models) {
    final groups = <String, List<AiModelVO>>{};
    for (final model in models) {
      final label = model.taskTags?.firstOrNull ?? '其他';
      (groups[label] ??= <AiModelVO>[]).add(model);
    }
    // 重新插入即移到 map 末尾
    final other = groups.remove('其他');
    if (other != null) {
      groups['其他'] = other;
    }
    return groups;
  }

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
        Text(
          '参数调节',
          style: theme.textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 16),
        _buildSlider(
          '去雾强度',
          _strength,
          0,
          100,
          (v) => setState(() => _strength = v),
        ),
        const SizedBox(height: 12),
        _buildSlider(
          '饱和度',
          _saturation,
          0,
          200,
          (v) => setState(() => _saturation = v),
        ),
        const SizedBox(height: 12),
        _buildSlider(
          '对比度',
          _contrast,
          0,
          200,
          (v) => setState(() => _contrast = v),
        ),
        const SizedBox(height: 12),
        _buildSlider(
          '锐化',
          _sharpen,
          0,
          100,
          (v) => setState(() => _sharpen = v),
        ),
      ],
    ),
  );

  Widget _buildSlider(
    String label,
    double value,
    double min,
    double max,
    ValueChanged<double> onChanged,
  ) => Row(
    children: [
      SizedBox(width: 64, child: Text(label)),
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
        child: Text(value.toStringAsFixed(0), textAlign: TextAlign.right),
      ),
    ],
  );

  Widget _buildProcessingSection(ThemeData theme, ProcessingState state) {
    switch (state.status) {
      case ProcessingStatus.idle:
        return FilledButton.icon(
          onPressed: state.canProcess ? () => _startProcessing(ref) : null,
          icon: const Icon(Icons.play_arrow),
          label: const Text('开始去雾处理'),
          style: FilledButton.styleFrom(
            minimumSize: const Size(double.infinity, 52),
          ),
        );

      case ProcessingStatus.processing:
        return Column(
          children: [
            _buildProcessingIndicator(state),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: () => ref.read(processingProvider.notifier).cancel(),
              icon: const Icon(Icons.stop_circle_outlined),
              label: const Text('取消处理'),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size(double.infinity, 48),
              ),
            ),
          ],
        );

      case ProcessingStatus.success:
        return Column(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.techGreen.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(AppTheme.radiusL),
                border: Border.all(
                  color: AppTheme.techGreen.withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                children: [
                  Icon(Icons.check_circle, color: AppTheme.techGreen, size: 24),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '处理完成!',
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: AppTheme.techGreen,
                          ),
                        ),
                        if (state.taskResult?.time != null)
                          Text(
                            '耗时: ${state.taskResult!.time}s',
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
            border: Border.all(
              color: theme.colorScheme.error.withValues(alpha: 0.3),
            ),
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
    final modelId = ref.read(processingProvider).selectedModel?.modelId;
    if (modelId != null) {
      ref
          .read(sharedPreferencesProvider)
          .setString(StorageConstants.lastImageModelId, modelId);
    }
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
