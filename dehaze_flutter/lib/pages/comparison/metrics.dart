import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/evaluation_model.dart';
import '../../providers/processing_provider.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../services/evaluation_service.dart';
import '../../theme/app_theme.dart';

/// 指标评估页面
///
/// 调用评估 API 计算 PSNR/SSIM/MSE/FSIM/LPIPS 指标
class MetricsPage extends ConsumerStatefulWidget {
  const MetricsPage({super.key});

  @override
  ConsumerState<MetricsPage> createState() => _MetricsPageState();
}

class _MetricsPageState extends ConsumerState<MetricsPage> {
  EvaluationMetrics? _metrics;
  bool _isEvaluating = false;
  String? _errorMessage;
  bool _noGtReference = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _evaluate());
  }

  Future<void> _evaluate() async {
    final state = ref.read(processingProvider);
    final predUrl = state.predictionResult?.resultUrl;
    final algorithmId = state.selectedAlgorithm?.id;
    final gtUrl = state.selectedImage?.cleanUrl;

    if (predUrl == null || algorithmId == null) {
      setState(() => _errorMessage = '缺少预测结果或算法信息');
      return;
    }

    // 无 GT 参考图（上传/拍照图片）时无法评估，PSNR/SSIM 等指标无意义
    if (gtUrl == null) {
      setState(() {
        _noGtReference = true;
        _isEvaluating = false;
        _errorMessage = null;
      });
      return;
    }

    setState(() {
      _isEvaluating = true;
      _errorMessage = null;
      _noGtReference = false;
    });

    try {
      final service = EvaluationService(ref.read(dioClientProvider));
      final request = EvaluationRequest(
        algorithmId: algorithmId,
        predUrl: predUrl,
        gtUrl: gtUrl,
      );
      // 评估为同步接口，直接返回指标
      final result = await service.evaluate(request);
      setState(() {
        _metrics = result.metricsModel;
        _isEvaluating = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = e.toString().replaceFirst('Exception: ', '');
        _isEvaluating = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(theme),
          Expanded(child: _buildBody(theme)),
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
            Icon(Icons.bar_chart_outlined, color: AppTheme.brandBlue),
            const SizedBox(width: 8),
            Text('指标评估', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          ],
        ),
      );

  Widget _buildBody(ThemeData theme) {
    if (_isEvaluating) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text('正在计算评估指标...', style: theme.textTheme.bodyLarge),
          ],
        ),
      );
    }

    // 无 GT 参考图：上传/拍照图片无法进行有意义的指标评估
    if (_noGtReference) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.block_outlined, size: 64, color: theme.colorScheme.onSurfaceVariant),
              const SizedBox(height: 16),
              Text('无法评估', style: theme.textTheme.titleLarge),
              const SizedBox(height: 8),
              Text(
                '当前图片无 GT 参考，无法评估。\n请使用数据集样例图片进行评估。',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_errorMessage != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 64, color: theme.colorScheme.error),
            const SizedBox(height: 16),
            Text('评估失败', style: theme.textTheme.titleLarge),
            const SizedBox(height: 8),
            Text(_errorMessage!, textAlign: TextAlign.center),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _evaluate,
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );
    }

    if (_metrics == null) {
      return const Center(child: Text('暂无评估数据'));
    }

    final items = _metrics!.toList();
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: items.length,
      itemBuilder: (context, index) => _MetricCard(item: items[index]),
    );
  }

  Widget _buildBottomNav(BuildContext context) => Container(
        padding: const EdgeInsets.all(12),
        child: Wrap(
          alignment: WrapAlignment.center,
          spacing: 8,
          children: [
            ActionChip(label: const Text('并排对比'), onPressed: () => context.go(AppRouterConfig.sideBySide)),
            ActionChip(label: const Text('重叠对比'), onPressed: () => context.go(AppRouterConfig.overlay)),
            ActionChip(label: const Text('放大镜'), onPressed: () => context.go(AppRouterConfig.magnifier)),
            ActionChip(label: const Text('滤镜调节'), onPressed: () => context.go(AppRouterConfig.filter)),
            ActionChip(label: const Text('算法信息'), onPressed: () => context.go(AppRouterConfig.algorithm)),
          ],
        ),
      );
}

/// 指标卡片
class _MetricCard extends StatelessWidget {
  const _MetricCard({required this.item});
  final MetricItem item;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = item.higherIsBetter ? AppTheme.techGreen : AppTheme.errorColor;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(Icons.analytics_outlined, color: color),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(item.name, style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: color.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          item.higherIsBetter ? '越高越好' : '越低越好',
                          style: TextStyle(fontSize: 10, color: color, fontWeight: FontWeight.w500),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(item.description, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                ],
              ),
            ),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  item.displayValue,
                  style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700, color: color),
                ),
                if (item.unit.isNotEmpty)
                  Text(item.unit, style: theme.textTheme.bodySmall),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
