import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/network/api_result.dart';
import '../../models/algorithm_model.dart';
import '../../models/prediction_model.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

/// 批量处理页面（L2，ToolsStack 内）
///
/// 支持：批量上传（≤20张）、算法选择、批量预测+轮询、进度+结果列表。
/// 移动端：垂直流程 / 桌面端：左右布局
class BatchPage extends ConsumerStatefulWidget {
  const BatchPage({super.key});

  @override
  ConsumerState<BatchPage> createState() => _BatchPageState();
}

enum BatchStep { upload, selectAlgorithm, processing, done }

class _BatchTask {
  const _BatchTask({
    required this.fileName,
    required this.fileId,
    required this.status,
    this.resultUrl,
    this.errorMessage,
    this.time,
  });

  final String fileName;
  final int fileId;
  final TaskStatus status;
  final String? resultUrl;
  final String? errorMessage;
  final int? time;

  _BatchTask copyWith({
    TaskStatus? status,
    String? resultUrl,
    String? errorMessage,
    int? time,
  }) =>
      _BatchTask(
        fileName: fileName,
        fileId: fileId,
        status: status ?? this.status,
        resultUrl: resultUrl ?? this.resultUrl,
        errorMessage: errorMessage,
        time: time ?? this.time,
      );
}

class _BatchPageState extends ConsumerState<BatchPage> {
  BatchStep _step = BatchStep.upload;
  final List<_BatchTask> _tasks = [];
  List<AlgorithmModel> _algorithms = [];
  AlgorithmModel? _selectedAlgorithm;
  bool _isLoadingAlgorithms = false;
  String? _algorithmError;

  // 进度追踪
  int _completedCount = 0;
  int _failedCount = 0;

  // 图片选择与上传
  final ImagePicker _picker = ImagePicker();
  final List<String> _uploadedFiles = [];
  final List<XFile> _pendingXFiles = [];
  bool _isUploading = false;

  @override
  Widget build(BuildContext context) {
    final isWide = ResponsiveUtils.isWideScreen(context);
    final theme = Theme.of(context);

    return Scaffold(
      body: ResponsiveConstraints(
        maxWidth: 1200,
        child: Column(
          children: [
            _buildHeader(theme),
            Expanded(
              child: isWide
                  ? _buildWideLayout(theme)
                  : _buildNarrowLayout(theme),
            ),
            if (_step == BatchStep.selectAlgorithm && _selectedAlgorithm != null)
              _buildBottomBar(theme),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) => Container(
        width: double.infinity,
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
                Icon(Icons.batch_prediction_outlined,
                    color: AppTheme.indigo, size: 24),
                const SizedBox(width: 8),
                Text(
                  '批量处理',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '一次上传多张图片，使用同一算法批量处理',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            _buildStepIndicator(theme),
          ],
        ),
      );

  Widget _buildStepIndicator(ThemeData theme) {
    const steps = ['上传图片', '选择算法', '批量处理', '查看结果'];
    final currentIdx = _step.index;
    final isWide = ResponsiveUtils.isWideScreen(context);

    return Row(
      children: List.generate(steps.length, (i) {
        final isActive = i == currentIdx;
        final isDone = i < currentIdx;

        return Expanded(
          child: Row(
            children: [
              if (i > 0)
                Expanded(
                  child: Container(
                    height: 2,
                    color: isDone
                        ? AppTheme.brandBlue
                        : theme.colorScheme.outline,
                  ),
                ),
              Container(
                width: 24,
                height: 24,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isActive || isDone
                      ? AppTheme.brandBlue
                      : theme.colorScheme.surfaceContainerHighest,
                ),
                child: Center(
                  child: isDone
                      ? const Icon(Icons.check, size: 14, color: Colors.white)
                      : Text(
                          '${i + 1}',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: isActive
                                ? Colors.white
                                : theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                ),
              ),
              const SizedBox(width: 4),
              if (isWide)
                Flexible(
                  child: Text(
                    steps[i],
                    style: TextStyle(
                      fontSize: 11,
                      color: isActive
                          ? AppTheme.brandBlue
                          : theme.colorScheme.onSurfaceVariant,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
            ],
          ),
        );
      }),
    );
  }

  /// 宽屏：左右布局
  Widget _buildWideLayout(ThemeData theme) => Padding(
        padding: ResponsiveUtils.getResponsivePadding(context),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              flex: 3,
              child: _buildCurrentStepContent(theme),
            ),
            const SizedBox(width: 24),
            Expanded(
              flex: 2,
              child: _buildSidePanel(theme),
            ),
          ],
        ),
      );

  /// 窄屏：垂直布局
  Widget _buildNarrowLayout(ThemeData theme) => SingleChildScrollView(
        padding: ResponsiveUtils.getResponsivePadding(context),
        child: Column(
          children: [
            _buildCurrentStepContent(theme),
            if (_step == BatchStep.processing || _step == BatchStep.done) ...[
              const SizedBox(height: 24),
              _buildSidePanel(theme),
            ],
          ],
        ),
      );

  Widget _buildCurrentStepContent(ThemeData theme) {
    switch (_step) {
      case BatchStep.upload:
        return _buildUploadStep(theme);
      case BatchStep.selectAlgorithm:
        return _buildAlgorithmStep(theme);
      case BatchStep.processing:
        return _buildProcessingStep(theme);
      case BatchStep.done:
        return _buildResultStep(theme);
    }
  }

  Widget _buildSidePanel(ThemeData theme) => Container(
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
                Icon(Icons.info_outline,
                    size: 18,
                    color: theme.colorScheme.onSurfaceVariant),
                const SizedBox(width: 8),
                Text(
                  '任务概览',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _infoRow(theme, '图片数量', '${_tasks.length}'),
            _infoRow(
              theme,
              '已完成',
              '$_completedCount',
              valueColor: AppTheme.techGreen,
            ),
            _infoRow(
              theme,
              '失败',
              '$_failedCount',
              valueColor: _failedCount > 0 ? AppTheme.errorColor : null,
            ),
            if (_selectedAlgorithm != null) ...[
              const SizedBox(height: 12),
              _infoRow(theme, '算法', _selectedAlgorithm!.name),
            ],
            if (_step == BatchStep.processing && _tasks.isNotEmpty) ...[
              const SizedBox(height: 16),
              LinearProgressIndicator(
                value: _tasks.isNotEmpty
                    ? (_completedCount + _failedCount) / _tasks.length
                    : 0,
              ),
            ],
          ],
        ),
      );

  Widget _infoRow(ThemeData theme, String label, String value,
          {Color? valueColor}) =>
      Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                )),
            Text(value,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: valueColor,
                )),
          ],
        ),
      );

  // ==================== Step: Upload ====================

  Widget _buildUploadStep(ThemeData theme) => Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
          border: Border.all(color: theme.colorScheme.outline),
        ),
        child: Column(
          children: [
            Icon(Icons.cloud_upload_outlined,
                size: 64, color: theme.colorScheme.onSurfaceVariant),
            const SizedBox(height: 16),
            Text(
              '点击或拖拽上传图片（≤20张）',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '支持 JPG、PNG 格式',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 24),
            // 选择/上传图片区域
            if (_isUploading)
              const Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 12),
                  Text('正在上传...'),
                ],
              )
            else
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  OutlinedButton.icon(
                    onPressed: _pickImages,
                    icon: const Icon(Icons.add_photo_alternate),
                    label: const Text('添加图片'),
                  ),
                  const SizedBox(width: 12),
                  if (_uploadedFiles.isNotEmpty)
                    FilledButton.icon(
                      onPressed: _initTasksFromUploads,
                      icon: const Icon(Icons.arrow_forward),
                      label: Text('已选 ${_uploadedFiles.length} 张，下一步'),
                    ),
                ],
              ),
            if (_uploadedFiles.isNotEmpty) ...[
              const SizedBox(height: 16),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _uploadedFiles
                    .map((f) => Chip(
                          label: Text(f, style: const TextStyle(fontSize: 12)),
                          deleteIcon: const Icon(Icons.close, size: 16),
                          onDeleted: () =>
                              setState(() {
                                final idx = _uploadedFiles.indexOf(f);
                                _uploadedFiles.removeAt(idx);
                                _pendingXFiles.removeAt(idx);
                              }),
                        ))
                    .toList(),
              ),
            ],
          ],
        ),
      );

  Future<void> _pickImages() async {
    try {
      final picked = await _picker.pickMultiImage(imageQuality: 80);
      if (picked.isEmpty) return;
      if (picked.length + _uploadedFiles.length > 20) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('最多上传20张图片')),
          );
        }
        return;
      }
      setState(() {
        for (final xfile in picked) {
          _uploadedFiles.add(xfile.name);
          _pendingXFiles.add(xfile);
        }
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('选择图片失败: ${e.toString()}')),
        );
      }
    }
  }

  Future<void> _initTasksFromUploads() async {
    setState(() => _isUploading = true);
    _tasks.clear();

    final fileService = ref.read(fileServiceProvider);
    for (int i = 0; i < _pendingXFiles.length; i++) {
      try {
        final xfile = _pendingXFiles[i];
        final bytes = await xfile.readAsBytes();
        final fileInfo = await fileService.uploadBytes(bytes, xfile.name);
        _tasks.add(_BatchTask(
          fileName: xfile.name,
          fileId: fileInfo.id,
          status: TaskStatus.processing,
        ));
      } catch (e) {
        _tasks.add(_BatchTask(
          fileName: _pendingXFiles[i].name,
          fileId: -1,
          status: TaskStatus.failed,
          errorMessage: '上传失败: ${e.toString()}',
        ));
      }
    }

    if (mounted) {
      setState(() {
        _isUploading = false;
        _step = BatchStep.selectAlgorithm;
      });
      _loadAlgorithms();
    }
  }

  // ==================== Step: Select Algorithm ====================

  Future<void> _loadAlgorithms() async {
    setState(() {
      _isLoadingAlgorithms = true;
      _algorithmError = null;
    });
    try {
      final service = ref.read(algorithmServiceProvider);
      final algorithms = await service.getList();
      final flatAlgorithms = algorithms.flatPublishedLeaves;
      if (mounted) {
        setState(() {
          _algorithms = flatAlgorithms;
          _isLoadingAlgorithms = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _algorithmError = extractErrorMessage(e);
          _isLoadingAlgorithms = false;
        });
      }
    }
  }

  Widget _buildAlgorithmStep(ThemeData theme) {
    if (_isLoadingAlgorithms) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_algorithmError != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline,
                size: 48, color: theme.colorScheme.error),
            const SizedBox(height: 8),
            Text(_algorithmError!),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _loadAlgorithms,
              icon: const Icon(Icons.refresh),
              label: const Text('重试'),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.only(bottom: 16),
      itemCount: _algorithms.length,
      itemBuilder: (context, index) {
        final algo = _algorithms[index];
        final isSelected = _selectedAlgorithm?.id == algo.id;

        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Material(
            color: isSelected
                ? AppTheme.brandBlue.withValues(alpha: 0.05)
                : theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
            child: InkWell(
              onTap: () => setState(() => _selectedAlgorithm = algo),
              borderRadius: BorderRadius.circular(AppTheme.radiusM),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                  border: Border.all(
                    color: isSelected
                        ? AppTheme.brandBlue
                        : theme.colorScheme.outline,
                    width: isSelected ? 2 : 1,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      algo.isDeepLearning
                          ? Icons.memory
                          : Icons.auto_fix_high,
                      color: isSelected
                          ? AppTheme.brandBlue
                          : theme.colorScheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(algo.name,
                              style: theme.textTheme.titleSmall?.copyWith(
                                fontWeight: FontWeight.w600,
                              )),
                          Text(algo.type,
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: theme.colorScheme.onSurfaceVariant,
                              )),
                        ],
                      ),
                    ),
                    if (isSelected)
                      Icon(Icons.check_circle,
                          color: AppTheme.brandBlue, size: 20),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildBottomBar(ThemeData theme) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          border: Border(top: BorderSide(color: theme.dividerColor)),
        ),
        child: SafeArea(
          child: Row(
            children: [
              Expanded(
                child: Text(
                  '已选算法: ${_selectedAlgorithm!.name}',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              FilledButton.icon(
                onPressed: _startBatchProcessing,
                icon: const Icon(Icons.play_arrow),
                label: Text('开始处理 (${_tasks.length} 张)'),
              ),
            ],
          ),
        ),
      );

  // ==================== Step: Processing ====================

  void _startBatchProcessing() {
    if (_selectedAlgorithm == null || _tasks.isEmpty) return;

    setState(() {
      _step = BatchStep.processing;
      _completedCount = 0;
      _failedCount = 0;
    });

    _processNextTask(0);
  }

  Future<void> _processNextTask(int index) async {
    if (index >= _tasks.length) {
      if (mounted) setState(() => _step = BatchStep.done);
      return;
    }

    try {
      final predictionService = ref.read(predictionServiceProvider);
      final startTime = DateTime.now();
      final request = PredictionRequest(
        algorithmId: _selectedAlgorithm!.id,
        fileId: _tasks[index].fileId,
      );
      final response = await predictionService.predictAndWait(request);
      final elapsed = DateTime.now().difference(startTime).inMilliseconds;

      if (!mounted) return;

      setState(() {
        _tasks[index] = _tasks[index].copyWith(
          status: response.status,
          resultUrl: response.resultUrl,
          errorMessage: response.errorMessage,
          time: elapsed,
        );
        if (response.status == TaskStatus.completed) {
          _completedCount++;
        } else {
          _failedCount++;
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _tasks[index] = _tasks[index].copyWith(
          status: TaskStatus.failed,
          errorMessage: extractErrorMessage(e),
        );
        _failedCount++;
      });
    }

    _processNextTask(index + 1);
  }

  Widget _buildProcessingStep(ThemeData theme) => Column(
        children: [
          const SizedBox(height: 24),
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            '正在批量处理... ($_completedCount/${_tasks.length})',
            style: theme.textTheme.titleMedium,
          ),
          const SizedBox(height: 24),
          Expanded(
            child: ListView.builder(
              itemCount: _tasks.length,
              itemBuilder: (context, index) {
                final task = _tasks[index];
                return ListTile(
                  leading: _taskStatusIcon(task.status),
                  title: Text(task.fileName,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w500,
                      )),
                  trailing: task.status == TaskStatus.processing
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : null,
                );
              },
            ),
          ),
        ],
      );

  // ==================== Step: Done ====================

  Widget _buildResultStep(ThemeData theme) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.techGreen.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(AppTheme.radiusL),
              border:
                  Border.all(color: AppTheme.techGreen.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                Icon(Icons.check_circle, color: AppTheme.techGreen, size: 24),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    '批量处理完成! 成功 $_completedCount 张，失败 $_failedCount 张',
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
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
                  onPressed: () => setState(() {
                    _step = BatchStep.upload;
                    _tasks.clear();
                    _uploadedFiles.clear();
                    _pendingXFiles.clear();
                    _completedCount = 0;
                    _failedCount = 0;
                  }),
                  icon: const Icon(Icons.refresh),
                  label: const Text('重新批量处理'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            '处理结果',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: ListView.builder(
              itemCount: _tasks.length,
              itemBuilder: (context, index) {
                final task = _tasks[index];
                return Card(
                  margin: const EdgeInsets.only(bottom: 8),
                  child: ListTile(
                    leading: _taskStatusIcon(task.status),
                    title: Text(task.fileName),
                    subtitle: task.status == TaskStatus.completed
                        ? Text('耗时: ${(task.time ?? 0) ~/ 1000}s')
                        : Text(task.errorMessage ?? '',
                            style: TextStyle(color: AppTheme.errorColor)),
                    trailing: task.status == TaskStatus.completed
                        ? Icon(Icons.visibility,
                            color: AppTheme.brandBlue, size: 20)
                        : null,
                  ),
                );
              },
            ),
          ),
        ],
      );

  Widget _taskStatusIcon(TaskStatus status) {
    switch (status) {
      case TaskStatus.processing:
        return Icon(Icons.hourglass_top,
            color: AppTheme.warningColor, size: 24);
      case TaskStatus.completed:
        return Icon(Icons.check_circle,
            color: AppTheme.techGreen, size: 24);
      case TaskStatus.failed:
        return Icon(Icons.error, color: AppTheme.errorColor, size: 24);
    }
  }
}
