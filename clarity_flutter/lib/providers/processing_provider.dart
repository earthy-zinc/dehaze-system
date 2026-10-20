import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/network/api_result.dart';
import '../core/network/task_poller.dart';
import '../models/ai_model_model.dart';
import '../models/dehaze_params.dart';
import '../models/image_task_model.dart';
import '../providers/providers.dart';
import '../services/image_task_service.dart';

/// 图像编辑模型类型（统一模型注册表分类）
const String imageEditModelType = 'image_edit';

/// 处理流程状态
class ProcessingState {
  const ProcessingState({
    this.status = ProcessingStatus.idle,
    this.selectedImage,
    this.selectedModel,
    this.taskResult,
    this.errorMessage,
    this.processingStartTime,
  });

  /// 当前选中的图片（含文件 ID）
  final SelectedImage? selectedImage;

  /// 当前选中的图像模型
  final AiModelVO? selectedModel;

  /// 图像任务结果
  final ImageTaskResponse? taskResult;

  /// 处理状态
  final ProcessingStatus status;

  /// 错误信息
  final String? errorMessage;

  /// 处理开始时间（用于计算已用时间），仅在 processing 状态下有值
  final DateTime? processingStartTime;

  ProcessingState copyWith({
    ProcessingStatus? status,
    SelectedImage? selectedImage,
    AiModelVO? selectedModel,
    ImageTaskResponse? taskResult,
    String? errorMessage,
    DateTime? processingStartTime,
    bool clearImage = false,
    bool clearModel = false,
    bool clearResult = false,
    bool clearProcessingStartTime = false,
  }) => ProcessingState(
    status: status ?? this.status,
    selectedImage: clearImage ? null : (selectedImage ?? this.selectedImage),
    selectedModel: clearModel ? null : (selectedModel ?? this.selectedModel),
    taskResult: clearResult ? null : (taskResult ?? this.taskResult),
    errorMessage: errorMessage,
    processingStartTime: clearProcessingStartTime
        ? null
        : (processingStartTime ?? this.processingStartTime),
  );

  /// 是否可以开始处理
  bool get canProcess => selectedImage != null && selectedModel != null;

  /// 是否有处理结果
  bool get hasResult => taskResult?.hasResult ?? false;
}

/// 处理流程状态枚举
enum ProcessingStatus { idle, processing, success, error }

/// 选中的图片（含文件 ID）
class SelectedImage {
  const SelectedImage({
    required this.fileId,
    required this.fileUrl,
    required this.fileName,
    this.bytes,
    this.cleanUrl,
  });

  /// 文件 ID（后端 SysFile.id）
  final int fileId;
  final String fileUrl;
  final String fileName;

  /// 原图字节流（内存态，跨平台渲染，Web 端不依赖文件路径）
  final Uint8List? bytes;

  /// 清晰图（Ground Truth）URL，用于指标评估
  /// 仅样例图片有值，上传/拍照图片为 null（无法评估）
  final String? cleanUrl;
}

/// 可用图像编辑模型列表
///
/// 数据源为统一模型注册表的启用模型（按登录用户 VIP 等级过滤）。
final imageEditModelsProvider = FutureProvider<List<AiModelVO>>((ref) {
  return ref
      .watch(aiModelServiceProvider)
      .listEnabledModels(modelType: imageEditModelType);
});

/// 处理流程状态管理
///
/// 后端图像任务为异步任务：POST 立即返回 status=processing，
/// 通过 submitAndWait 轮询 GET 直至终态。
class ProcessingNotifier extends StateNotifier<ProcessingState> {
  ProcessingNotifier(this._imageTaskService) : super(const ProcessingState());

  final ImageTaskService _imageTaskService;

  PollCancellation? _cancellation;
  int? _taskId;

  /// 设置选中的图片
  void setSelectedImage(SelectedImage image) {
    state = state.copyWith(
      selectedImage: image,
      clearResult: true,
      status: ProcessingStatus.idle,
      errorMessage: null,
    );
  }

  /// 设置选中的图像模型
  void setSelectedModel(AiModelVO model) {
    state = state.copyWith(
      selectedModel: model,
      clearResult: true,
      status: ProcessingStatus.idle,
      errorMessage: null,
    );
  }

  /// 执行去雾处理
  Future<void> process({DehazeParams? params}) async {
    if (!state.canProcess) {
      state = state.copyWith(
        errorMessage: '请先选择图片和模型',
        status: ProcessingStatus.error,
        clearProcessingStartTime: true,
      );
      return;
    }

    state = state.copyWith(
      status: ProcessingStatus.processing,
      errorMessage: null,
      clearResult: true,
      processingStartTime: DateTime.now(),
    );

    final cancellation = PollCancellation();
    _cancellation = cancellation;
    _taskId = null;

    try {
      final request = ImageEditRequest(
        model: state.selectedModel!.modelId,
        fileId: state.selectedImage!.fileId,
        // 仅在非默认参数时传值，避免干扰模型默认行为
        params: params?.isDefault == false ? params!.toJson() : null,
      );

      final response = await _imageTaskService.submitAndWait(
        request,
        options: PollOptions(cancellation: cancellation),
        onSubmitted: (taskId) => _taskId = taskId,
      );

      if (response.status == TaskStatus.failed) {
        state = state.copyWith(
          status: ProcessingStatus.error,
          errorMessage: response.errorMessage ?? '处理失败',
          clearProcessingStartTime: true,
        );
      } else if (response.hasResult) {
        state = state.copyWith(
          taskResult: response,
          status: ProcessingStatus.success,
          clearProcessingStartTime: true,
        );
      } else {
        state = state.copyWith(
          status: ProcessingStatus.error,
          errorMessage: '处理未返回结果，请重试',
          clearProcessingStartTime: true,
        );
      }
    } on PollCancelledException {
      // 用户主动取消：不视为错误
      state = state.copyWith(
        status: ProcessingStatus.idle,
        errorMessage: null,
        clearProcessingStartTime: true,
      );
    } catch (e) {
      state = state.copyWith(
        status: ProcessingStatus.error,
        errorMessage: extractErrorMessage(e),
        clearProcessingStartTime: true,
      );
    } finally {
      _cancellation = null;
      _taskId = null;
    }
  }

  /// 取消正在进行的处理（本地停止轮询并请求后端取消任务）
  Future<void> cancel() async {
    if (state.status != ProcessingStatus.processing) return;
    _cancellation?.cancel();
    final taskId = _taskId;
    if (taskId != null) {
      try {
        await _imageTaskService.cancelTask(taskId);
      } catch (_) {
        // 后端取消失败不影响本地停止轮询
      }
    }
    state = state.copyWith(
      status: ProcessingStatus.idle,
      errorMessage: null,
      clearProcessingStartTime: true,
    );
  }

  /// 重置状态
  void reset() {
    state = const ProcessingState();
  }

  /// 清除错误
  void clearError() {
    state = state.copyWith(
      errorMessage: null,
      status: ProcessingStatus.idle,
      clearProcessingStartTime: true,
    );
  }
}

// ==================== Providers ====================

/// 处理流程 Provider
final processingProvider =
    StateNotifierProvider<ProcessingNotifier, ProcessingState>((ref) {
      final imageTaskService = ref.watch(imageTaskServiceProvider);
      return ProcessingNotifier(imageTaskService);
    });
