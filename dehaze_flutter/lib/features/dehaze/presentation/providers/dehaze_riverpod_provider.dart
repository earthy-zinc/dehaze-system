import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../domain/entities/dehaze_image.dart';
import '../../domain/repositories/dehaze_repository.dart';
import '../../../../services/providers.dart';

// 状态类
class DehazeState {
  final List<DehazeImage> history;
  final bool isLoading;
  final bool isProcessing;
  final String? errorMessage;
  final DehazeImage? currentProcessingImage;
  final List<DehazeAlgorithm> availableAlgorithms;
  final DehazeParameters currentParameters;

  const DehazeState({
    this.history = const [],
    this.isLoading = false,
    this.isProcessing = false,
    this.errorMessage,
    this.currentProcessingImage,
    this.availableAlgorithms = const [
      DehazeAlgorithm.darkChannel,
      DehazeAlgorithm.atmosphericLight,
      DehazeAlgorithm.retinex,
      DehazeAlgorithm.colorAttenuation,
    ],
    this.currentParameters = const DehazeParameters(),
  });

  DehazeState copyWith({
    List<DehazeImage>? history,
    bool? isLoading,
    bool? isProcessing,
    String? errorMessage,
    DehazeImage? currentProcessingImage,
    List<DehazeAlgorithm>? availableAlgorithms,
    DehazeParameters? currentParameters,
  }) {
    return DehazeState(
      history: history ?? this.history,
      isLoading: isLoading ?? this.isLoading,
      isProcessing: isProcessing ?? this.isProcessing,
      errorMessage: errorMessage ?? this.errorMessage,
      currentProcessingImage: currentProcessingImage ?? this.currentProcessingImage,
      availableAlgorithms: availableAlgorithms ?? this.availableAlgorithms,
      currentParameters: currentParameters ?? this.currentParameters,
    );
  }
}

// Riverpod StateNotifier
class DehazeNotifier extends StateNotifier<DehazeState> {
  final DehazeRepository _repository;

  DehazeNotifier({required DehazeRepository repository})
      : _repository = repository,
        super(const DehazeState());

  // 加载历史记录
  Future<void> loadHistory() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    final result = await _repository.getDehazeHistory();

    result.fold(
      (failure) {
        state = state.copyWith(
          isLoading: false,
          errorMessage: failure.message,
        );
      },
      (history) {
        state = state.copyWith(
          isLoading: false,
          history: history,
        );
      },
    );
  }

  // 加载可用算法
  Future<void> loadAvailableAlgorithms() async {
    final result = await _repository.getAvailableAlgorithms();

    result.fold(
      (failure) {
        // 如果获取算法失败，使用默认算法
        state = state.copyWith(
          errorMessage: failure.message,
        );
      },
      (algorithms) {
        state = state.copyWith(
          availableAlgorithms: algorithms,
        );
      },
    );
  }

  // 处理图像
  Future<void> processImage(String imagePath) async {
    state = state.copyWith(isProcessing: true, errorMessage: null);

    // 参数验证（原UseCase中的逻辑）
    if (imagePath.isEmpty) {
      state = state.copyWith(
        isProcessing: false,
        errorMessage: 'Image path cannot be empty',
      );
      return;
    }

    final params = state.currentParameters;

    if (params.strength < 0.0 || params.strength > 1.0) {
      state = state.copyWith(
        isProcessing: false,
        errorMessage: 'Strength must be between 0.0 and 1.0',
      );
      return;
    }

    if (params.contrast < 0.0 || params.contrast > 3.0) {
      state = state.copyWith(
        isProcessing: false,
        errorMessage: 'Contrast must be between 0.0 and 3.0',
      );
      return;
    }

    if (params.brightness < 0.0 || params.brightness > 3.0) {
      state = state.copyWith(
        isProcessing: false,
        errorMessage: 'Brightness must be between 0.0 and 3.0',
      );
      return;
    }

    final result = await _repository.processImage(imagePath, params);

    result.fold(
      (failure) {
        state = state.copyWith(
          isProcessing: false,
          errorMessage: failure.message,
        );
      },
      (image) {
        state = state.copyWith(
          isProcessing: false,
          currentProcessingImage: image,
        );

        // 重新加载历史记录
        loadHistory();
      },
    );
  }

  // 删除图像
  Future<void> deleteImage(String imageId) async {
    state = state.copyWith(errorMessage: null);

    final result = await _repository.deleteDehazeImage(imageId);

    result.fold(
      (failure) {
        state = state.copyWith(errorMessage: failure.message);
      },
      (_) {
        // 重新加载历史记录
        loadHistory();
      },
    );
  }

  // 更新参数
  void updateParameters(DehazeParameters parameters) {
    state = state.copyWith(currentParameters: parameters);
  }

  // 清除当前处理的图像
  void clearCurrentProcessingImage() {
    state = state.copyWith(currentProcessingImage: null);
  }

  // 清除错误信息
  void clearError() {
    state = state.copyWith(errorMessage: null);
  }
}

// Riverpod Provider
final dehazeProvider = StateNotifierProvider<DehazeNotifier, DehazeState>((ref) {
  final repository = ref.read(dehazeRepositoryProvider);
  return DehazeNotifier(repository: repository);
});