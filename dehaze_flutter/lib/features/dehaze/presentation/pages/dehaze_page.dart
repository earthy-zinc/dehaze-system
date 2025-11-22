import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/dehaze_riverpod_provider.dart';
import '../widgets/dehaze_controls_widget.dart';
import '../widgets/dehaze_history_widget.dart';
import '../widgets/processing_status_widget.dart';

class DehazePage extends ConsumerStatefulWidget {
  const DehazePage({super.key});

  @override
  ConsumerState<DehazePage> createState() => _DehazePageState();
}

class _DehazePageState extends ConsumerState<DehazePage> {
  @override
  Widget build(BuildContext context) {
    final dehazeState = ref.watch(dehazeProvider);
    final dehazeNotifier = ref.read(dehazeProvider.notifier);

    // 初始化数据
    ref.listen(dehazeProvider, (previous, next) {
      if (previous == null &&
          (next.history.isEmpty || next.availableAlgorithms.isEmpty)) {
        dehazeNotifier.loadHistory();
        dehazeNotifier.loadAvailableAlgorithms();
      }
    });

    return Scaffold(
      appBar: AppBar(
        title: const Text('图像去雾'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: dehazeState.isLoading && dehazeState.history.isEmpty
          ? const Center(child: CircularProgressIndicator())
          : LayoutBuilder(
              builder: (context, constraints) {
                // 根据屏幕高度调整布局
                final isCompact = constraints.maxHeight < 600;

                return SingleChildScrollView(
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight,
                    ),
                    child: Column(
                      children: [
                        // 控制面板 - 动态高度
                        Container(
                          constraints: BoxConstraints(
                            maxHeight: isCompact ? 300 : 400,
                          ),
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: DehazeControlsWidget(
                              currentParameters: dehazeState.currentParameters,
                              availableAlgorithms:
                                  dehazeState.availableAlgorithms,
                              isProcessing: dehazeState.isProcessing,
                              onParametersChanged:
                                  dehazeNotifier.updateParameters,
                              onProcessImage: dehazeNotifier.processImage,
                            ),
                          ),
                        ),

                        // 处理状态 - 仅在需要时显示
                        if (dehazeState.isProcessing ||
                            dehazeState.currentProcessingImage != null)
                          Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                            ),
                            child: ProcessingStatusWidget(
                              isProcessing: dehazeState.isProcessing,
                              currentImage: dehazeState.currentProcessingImage,
                              onClearCurrent:
                                  dehazeNotifier.clearCurrentProcessingImage,
                            ),
                          ),

                        if (dehazeState.isProcessing ||
                            dehazeState.currentProcessingImage != null)
                          const SizedBox(height: 16),

                        // 分隔线
                        if (dehazeState.history.isNotEmpty)
                          const Padding(
                            padding: EdgeInsets.symmetric(horizontal: 16),
                            child: Divider(),
                          ),

                        // 历史记录 - 最小高度
                        if (dehazeState.history.isNotEmpty)
                          const SizedBox(height: 16),

                        if (dehazeState.history.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                            ),
                            child: DehazeHistoryWidget(
                              history: dehazeState.history,
                              onDeleteImage: dehazeNotifier.deleteImage,
                            ),
                          ),

                        // 底部间距
                        const SizedBox(height: 16),

                        // 显示错误信息
                        if (dehazeState.errorMessage != null)
                          Padding(
                            padding: const EdgeInsets.all(16),
                            child: Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.red.shade50,
                                border: Border.all(color: Colors.red.shade200),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Row(
                                children: [
                                  Icon(Icons.error, color: Colors.red.shade600),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      dehazeState.errorMessage!,
                                      style: TextStyle(
                                        color: Colors.red.shade600,
                                      ),
                                    ),
                                  ),
                                  IconButton(
                                    icon: const Icon(Icons.close),
                                    onPressed: dehazeNotifier.clearError,
                                  ),
                                ],
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}
