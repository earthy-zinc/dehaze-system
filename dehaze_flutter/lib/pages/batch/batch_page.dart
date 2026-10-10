import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../../core/network/api_result.dart';
import '../../models/ai_model_model.dart';
import '../../models/image_task_model.dart';
import '../../providers/processing_provider.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';

/// 会员接口不可用时的批量上限（以会员权益 benefits.batchLimit 为准）
const int defaultBatchLimit = 5;

/// 批量处理页面（L2，ToolsStack 内）
///
/// 一次上传多张图片，使用同一图像模型批量处理：
/// 逐张上传换 fileId → 逐张提交图像任务并轮询至终态。
/// 移动端：垂直流程 / 桌面端：左右布局
class BatchPage extends ConsumerStatefulWidget {
  const BatchPage({super.key});

  @override
  ConsumerState<BatchPage> createState() => _BatchPageState();
}

enum BatchStep { upload, selectModel, processing, done }

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

  /// 后端返回的处理耗时（秒）
  final int? time;

  _BatchTask copyWith({
    TaskStatus? status,
    String? resultUrl,
    String? errorMessage,
    int? time,
  }) => _BatchTask(
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
  AiModelVO? _selectedModel;
  int _maxImages = defaultBatchLimit;

  // 进度追踪
  int _completedCount = 0;
  int _failedCount = 0;

  // 图片选择与上传
  final ImagePicker _picker = ImagePicker();
  final List<String> _uploadedFiles = [];
  final List<XFile> _pendingXFiles = [];
  bool _isUploading = false;

  @override
  void initState() {
    super.initState();
    _loadBatchLimit();
  }

  Future<void> _loadBatchLimit() async {
    try {
      final profile = await ref.read(memberServiceProvider).getProfile();
      if (!mounted) return;
      setState(() => _maxImages = profile.benefits.batchLimit);
    } catch (_) {
      // 非会员或未登录时会员接口不可用，沿用默认上限
    }
  }

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
            Expanded(child: _buildAdaptiveLayout(theme, isWide)),
            if (_step == BatchStep.selectModel && _selectedModel != null)
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
      border: Border(bottom: BorderSide(color: theme.dividerColor, width: 1)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              Icons.batch_prediction_outlined,
              color: AppTheme.indigo,
              size: 24,
            ),
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
          '一次上传多张图片，使用同一模型批量处理',
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
    const steps = ['上传图片', '选择模型', '批量处理', '查看结果'];
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

  /// 宽屏：左右分栏（始终带侧栏）；窄屏：垂直滚动（仅处理中 / 完成显示侧栏）
  Widget _buildAdaptiveLayout(ThemeData theme, bool isWide) {
    final content = _buildCurrentStepContent(theme);
    final padding = ResponsiveUtils.getResponsivePadding(context);
    if (isWide) {
      return Padding(
        padding: padding,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(flex: 3, child: content),
            const SizedBox(width: 24),
            Expanded(flex: 2, child: _buildSidePanel(theme)),
          ],
        ),
      );
    }
    final showSidePanel =
        _step == BatchStep.processing || _step == BatchStep.done;
    return SingleChildScrollView(
      padding: padding,
      child: Column(
        children: [
          content,
          if (showSidePanel) ...[
            const SizedBox(height: 24),
            _buildSidePanel(theme),
          ],
        ],
      ),
    );
  }

  Widget _buildCurrentStepContent(ThemeData theme) {
    switch (_step) {
      case BatchStep.upload:
        return _buildUploadStep(theme);
      case BatchStep.selectModel:
        return _buildModelStep(theme);
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
            Icon(
              Icons.info_outline,
              size: 18,
              color: theme.colorScheme.onSurfaceVariant,
            ),
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
        if (_selectedModel != null) ...[
          const SizedBox(height: 12),
          _infoRow(theme, '模型', _selectedModel!.displayName),
        ],
        if (_step == BatchStep.processing && _tasks.isNotEmpty) ...[
          const SizedBox(height: 16),
          LinearProgressIndicator(
            value: (_completedCount + _failedCount) / _tasks.length,
          ),
        ],
      ],
    ),
  );

  Widget _infoRow(
    ThemeData theme,
    String label,
    String value, {
    Color? valueColor,
  }) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        Text(
          value,
          style: theme.textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: valueColor,
          ),
        ),
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
        Icon(
          Icons.cloud_upload_outlined,
          size: 64,
          color: theme.colorScheme.onSurfaceVariant,
        ),
        const SizedBox(height: 16),
        Text(
          '点击或拖拽上传图片（≤$_maxImages张）',
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
                .map(
                  (f) => Chip(
                    label: Text(f, style: const TextStyle(fontSize: 12)),
                    deleteIcon: const Icon(Icons.close, size: 16),
                    onDeleted: () => setState(() {
                      final idx = _uploadedFiles.indexOf(f);
                      _uploadedFiles.removeAt(idx);
                      _pendingXFiles.removeAt(idx);
                    }),
                  ),
                )
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
      if (picked.length + _uploadedFiles.length > _maxImages) {
        if (mounted) {
          ScaffoldMessenger.of(
            context,
          ).showSnackBar(SnackBar(content: Text('最多上传$_maxImages张图片')));
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
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('选择图片失败: ${e.toString()}')));
      }
    }
  }

  Future<void> _initTasksFromUploads() async {
    setState(() => _isUploading = true);
    _tasks.clear();

    final fileService = ref.read(fileServiceProvider);
    for (final xfile in _pendingXFiles) {
      try {
        final bytes = await xfile.readAsBytes();
        final fileInfo = await fileService.uploadBytes(bytes, xfile.name);
        _tasks.add(
          _BatchTask(
            fileName: xfile.name,
            fileId: fileInfo.id,
            status: TaskStatus.processing,
          ),
        );
      } catch (e) {
        _tasks.add(
          _BatchTask(
            fileName: xfile.name,
            fileId: -1,
            status: TaskStatus.failed,
            errorMessage: '上传失败: ${e.toString()}',
          ),
        );
      }
    }

    if (mounted) {
      setState(() {
        _isUploading = false;
        _step = BatchStep.selectModel;
      });
    }
  }

  // ==================== Step: Select Model ====================

  Widget _buildModelStep(ThemeData theme) => ref
      .watch(imageEditModelsProvider)
      .when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (_, _) => _buildModelError(theme),
        data: (models) => _buildModelList(theme, models),
      );

  Widget _buildModelError(ThemeData theme) => Center(
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
        const SizedBox(height: 8),
        const Text('加载图像模型失败'),
        const SizedBox(height: 12),
        ElevatedButton.icon(
          onPressed: () => ref.invalidate(imageEditModelsProvider),
          icon: const Icon(Icons.refresh),
          label: const Text('重试'),
        ),
      ],
    ),
  );

  Widget _buildModelList(ThemeData theme, List<AiModelVO> models) {
    if (models.isEmpty) {
      return const Center(child: Text('暂无可用图像模型'));
    }

    return ListView.builder(
      padding: const EdgeInsets.only(bottom: 16),
      itemCount: models.length,
      itemBuilder: (context, index) {
        final model = models[index];
        final isSelected = _selectedModel?.modelId == model.modelId;

        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Material(
            color: isSelected
                ? AppTheme.brandBlue.withValues(alpha: 0.05)
                : theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
            child: InkWell(
              onTap: () => setState(() => _selectedModel = model),
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
                      Icons.auto_fix_high,
                      color: isSelected
                          ? AppTheme.brandBlue
                          : theme.colorScheme.onSurfaceVariant,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            model.displayName,
                            style: theme.textTheme.titleSmall?.copyWith(
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            model.taskTags?.join('、') ?? model.modelType ?? '',
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (isSelected)
                      Icon(
                        Icons.check_circle,
                        color: AppTheme.brandBlue,
                        size: 20,
                      ),
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
              '已选模型: ${_selectedModel!.displayName}',
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
    if (_selectedModel == null || _tasks.isEmpty) return;

    setState(() {
      _step = BatchStep.processing;
      _completedCount = 0;
      _failedCount = _tasks.where((t) => t.status == TaskStatus.failed).length;
    });

    _processNextTask(0);
  }

  Future<void> _processNextTask(int index) async {
    // 页面已销毁则停止后续提交（避免离开页面后继续产生计费任务）
    if (!mounted) return;
    if (index >= _tasks.length) {
      if (mounted) setState(() => _step = BatchStep.done);
      return;
    }

    if (_tasks[index].status != TaskStatus.processing) {
      await _processNextTask(index + 1);
      return;
    }

    try {
      final request = ImageEditRequest(
        model: _selectedModel!.modelId,
        fileId: _tasks[index].fileId,
      );
      final response = await ref
          .read(imageTaskServiceProvider)
          .submitAndWait(request);

      if (!mounted) return;

      setState(() {
        _tasks[index] = _tasks[index].copyWith(
          status: response.status,
          resultUrl: response.resultUrl,
          errorMessage: response.errorMessage,
          time: response.time,
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

    await _processNextTask(index + 1);
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
              title: Text(
                task.fileName,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                ),
              ),
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
          border: Border.all(color: AppTheme.techGreen.withValues(alpha: 0.3)),
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
            final isCompleted = task.status == TaskStatus.completed;

            return Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ListTile(
                leading: _taskStatusIcon(task.status),
                title: Text(task.fileName),
                subtitle: isCompleted
                    ? Text('耗时: ${task.time ?? 0}s')
                    : Text(
                        task.errorMessage ?? '',
                        style: TextStyle(color: AppTheme.errorColor),
                      ),
                trailing: isCompleted
                    ? Icon(
                        Icons.visibility,
                        color: AppTheme.brandBlue,
                        size: 20,
                      )
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
        return Icon(
          Icons.hourglass_top,
          color: AppTheme.warningColor,
          size: 24,
        );
      case TaskStatus.completed:
        return Icon(Icons.check_circle, color: AppTheme.techGreen, size: 24);
      case TaskStatus.failed:
        return Icon(Icons.error, color: AppTheme.errorColor, size: 24);
      case TaskStatus.canceled:
        return Icon(Icons.cancel, color: AppTheme.warningColor, size: 24);
    }
  }
}
