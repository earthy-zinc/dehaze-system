import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/network/api_result.dart';
import '../models/algorithm_model.dart';
import '../models/dehaze_params.dart';
import '../models/prediction_model.dart';
import '../providers/providers.dart';
import '../services/algorithm_service.dart';
import '../services/file_service.dart';
import '../services/prediction_service.dart';

/// 处理流程状态
class ProcessingState {
  const ProcessingState({
    this.status = ProcessingStatus.idle,
    this.selectedImage,
    this.selectedAlgorithm,
    this.predictionResult,
    this.errorMessage,
    this.processingStartTime,
  });

  /// 当前选中的图片（含文件 ID）
  final SelectedImage? selectedImage;

  /// 当前选中的算法
  final AlgorithmModel? selectedAlgorithm;

  /// 预测结果
  final PredictionResponse? predictionResult;

  /// 处理状态
  final ProcessingStatus status;

  /// 错误信息
  final String? errorMessage;

  /// 处理开始时间（用于计算已用时间），仅在 processing 状态下有值
  final DateTime? processingStartTime;

  ProcessingState copyWith({
    ProcessingStatus? status,
    SelectedImage? selectedImage,
    AlgorithmModel? selectedAlgorithm,
    PredictionResponse? predictionResult,
    String? errorMessage,
    DateTime? processingStartTime,
    bool clearImage = false,
    bool clearAlgorithm = false,
    bool clearResult = false,
    bool clearProcessingStartTime = false,
  }) =>
      ProcessingState(
        status: status ?? this.status,
        selectedImage:
            clearImage ? null : (selectedImage ?? this.selectedImage),
        selectedAlgorithm: clearAlgorithm
            ? null
            : (selectedAlgorithm ?? this.selectedAlgorithm),
        predictionResult:
            clearResult ? null : (predictionResult ?? this.predictionResult),
        errorMessage: errorMessage,
        processingStartTime: clearProcessingStartTime
            ? null
            : (processingStartTime ?? this.processingStartTime),
      );

  /// 是否可以开始处理
  bool get canProcess => selectedImage != null && selectedAlgorithm != null;

  /// 是否有处理结果
  bool get hasResult => predictionResult?.hasResult ?? false;
}

/// 处理流程状态枚举
enum ProcessingStatus {
  idle,
  processing,
  success,
  error,
}

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

/// 处理流程状态管理
///
/// 后端预测为同步接口：POST /prediction 直接返回结果，无需轮询。
class ProcessingNotifier extends StateNotifier<ProcessingState> {
  ProcessingNotifier(this._predictionService)
      : super(const ProcessingState());

  final PredictionService _predictionService;

  /// 设置选中的图片
  void setSelectedImage(SelectedImage image) {
    state = state.copyWith(
      selectedImage: image,
      clearResult: true,
      status: ProcessingStatus.idle,
      errorMessage: null,
    );
  }

  /// 设置选中的算法
  void setSelectedAlgorithm(AlgorithmModel algorithm) {
    state = state.copyWith(
      selectedAlgorithm: algorithm,
      clearResult: true,
      status: ProcessingStatus.idle,
      errorMessage: null,
    );
  }

  /// 执行去雾处理
  Future<void> process({DehazeParams? params}) async {
    if (!state.canProcess) {
      state = state.copyWith(
        errorMessage: '请先选择图片和算法',
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

    try {
      final request = PredictionRequest(
        algorithmId: state.selectedAlgorithm!.id,
        fileId: state.selectedImage!.fileId,
        // 仅在非默认参数时传值，避免干扰算法默认行为
        params: params?.isDefault == false ? params!.toJson() : null,
      );

      final response = await _predictionService.predict(request);

      if (response.hasResult) {
        state = state.copyWith(
          predictionResult: response,
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
    } catch (e) {
      state = state.copyWith(
        status: ProcessingStatus.error,
        errorMessage: extractErrorMessage(e),
        clearProcessingStartTime: true,
      );
    }
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

/// 预测服务 Provider
final predictionServiceProvider = Provider<PredictionService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return PredictionService(dio);
});

/// 文件服务 Provider
final fileServiceProvider = Provider<FileService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return FileService(dio);
});

/// 算法服务 Provider
final algorithmServiceProvider = Provider<AlgorithmService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return AlgorithmService(dio);
});

/// 处理流程 Provider
final processingProvider =
    StateNotifierProvider<ProcessingNotifier, ProcessingState>((ref) {
  final predictionService = ref.watch(predictionServiceProvider);
  return ProcessingNotifier(predictionService);
});
