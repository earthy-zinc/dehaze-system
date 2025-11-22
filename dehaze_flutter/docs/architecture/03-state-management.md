# 状态管理架构设计

**文档版本**: v1.0
**最后更新**: 2025-11-22
**项目名称**: dehaze_flutter
**参考文档**: [架构设计](../design/02-architecture.md)、[UI组件设计](02-ui-components.md)

---

## 📋 概述

本文档详细描述了Flutter图像去雾系统的状态管理架构设计，基于[架构设计文档](../design/02-architecture.md)中的Bloc模式选择，专注于前端状态管理的详细实现方案。

---

## 🏗️ 状态管理架构概览

### 技术选型：Bloc + Cubit

基于[架构设计中的决策](../design/02-architecture.md#4-1-技术选型决策)，选择**Bloc**作为主要状态管理方案，结合**Cubit**用于简单场景。

#### 选择理由
- **清晰的业务逻辑分离**：Bloc强制将业务逻辑与UI分离
- **可测试性强**：状态转换逻辑易于单元测试
- **调试友好**：BlocObserver提供完整的状态变化日志
- **类型安全**：编译时类型检查，减少运行时错误
- **团队协作**：统一的状态管理模式，便于团队协作

### 架构层次

```
┌─────────────────────────────────────────┐
│              UI Layer                   │
│  ┌─────────────┬─────────────┐         │
│  │   Widgets   │  BuildContext│         │
│  └─────────────┴─────────────┘         │
└─────────────────────────────────────────┘
                    ↓ Events
┌─────────────────────────────────────────┐
│            Bloc/Cubit Layer              │
│  ┌─────────────┬─────────────┐         │
│  │   Events    │   States    │         │
│  └─────────────┴─────────────┘         │
└─────────────────────────────────────────┘
                    ↓ Use Cases
┌─────────────────────────────────────────┐
│            Business Logic               │
│  ┌─────────────┬─────────────┐         │
│  │  Use Cases  │ Repositories│         │
│  └─────────────┴─────────────┘         │
└─────────────────────────────────────────┘
                    ↓ Data Sources
┌─────────────────────────────────────────┐
│             Data Layer                   │
│  ┌─────────────┬─────────────┐         │
│  │   Remote    │    Local    │         │
│  │   APIs      │  Storage    │         │
│  └─────────────┴─────────────┘         │
└─────────────────────────────────────────┘
```

---

## 🔧 核心状态管理组件

### 全局状态配置

#### 服务定位器配置
```dart
// lib/core/di/service_locator.dart
import 'package:get_it/get_it.dart';
import 'package:bloc/bloc.dart';
import 'package:hydrated_bloc/hydrated_bloc.dart';

final GetIt serviceLocator = GetIt.instance;

Future<void> setupServiceLocator() async {
  // 状态管理器注册
  serviceLocator.registerLazySingleton(() => AppBlocObserver());
  serviceLocator.registerLazySingleton(() => HydratedStorage());

  // Repository注册
  serviceLocator.registerLazySingleton<ImageRepository>(
    () => ImageRepositoryImpl(),
  );
  serviceLocator.registerLazySingleton<AlgorithmRepository>(
    () => AlgorithmRepositoryImpl(),
  );
  serviceLocator.registerLazySingleton<ProcessingRepository>(
    () => ProcessingRepositoryImpl(),
  );

  // Service注册
  serviceLocator.registerLazySingleton<ApiService>(
    () => ApiService(),
  );
  serviceLocator.registerLazySingleton<StorageService>(
    () => StorageServiceImpl(),
  );

  // Bloc/Cubit注册
  serviceLocator.registerFactory<ImageInputCubit>(
    () => ImageInputCubit(serviceLocator<ImageRepository>()),
  );
  serviceLocator.registerFactory<AlgorithmSelectCubit>(
    () => AlgorithmSelectCubit(serviceLocator<AlgorithmRepository>()),
  );
  serviceLocator.registerFactory<ProcessingCubit>(
    () => ProcessingCubit(
      serviceLocator<ProcessingRepository>(),
      serviceLocator<ImageRepository>(),
    ),
  );
  serviceLocator.registerFactory<ComparisonCubit>(
    () => ComparisonCubit(),
  );
}
```

#### Bloc观察器
```dart
// lib/core/observers/app_bloc_observer.dart
class AppBlocObserver extends BlocObserver {
  @override
  void onCreate(BlocBase bloc) {
    super.onCreate(bloc);
    log('Bloc Created: ${bloc.runtimeType}');
  }

  @override
  void onEvent(Bloc bloc, Object? event) {
    super.onEvent(bloc, event);
    log('Event: ${bloc.runtimeType} -> $event');
  }

  @override
  void onChange(BlocBase bloc, Change change) {
    super.onChange(bloc, change);
    log('State Change: ${bloc.runtimeType} -> ${change.nextState}');
  }

  @override
  void onTransition(Bloc bloc, Transition transition) {
    super.onTransition(bloc, transition);
    log('Transition: ${bloc.runtimeType} -> ${transition.nextState}');
  }

  @override
  void onError(BlocBase bloc, Object error, StackTrace stackTrace) {
    super.onError(bloc, error, stackTrace);
    log('Error: ${bloc.runtimeType} -> $error', stackTrace: stackTrace);
  }

  @override
  void onClose(BlocBase bloc) {
    super.onClose(bloc);
    log('Bloc Closed: ${bloc.runtimeType}');
  }
}
```

---

## 📷 图像输入状态管理

### 状态定义
```dart
// lib/features/image_input/bloc/image_input_state.dart
part of 'image_input_cubit.dart';

enum ImageInputStatus {
  initial,       // 初始状态
  loading,       // 加载中
  success,       // 成功
  error,         // 错误
  processing,    // 处理中
}

class ImageInputState extends Equatable {
  final List<ImageFile> selectedImages;
  final ImageInputStatus status;
  final String? errorMessage;
  final int maxImages;
  final bool isSelecting;
  final Map<String, double> uploadProgress;

  const ImageInputState({
    this.selectedImages = const [],
    this.status = ImageInputStatus.initial,
    this.errorMessage,
    this.maxImages = 5,
    this.isSelecting = false,
    this.uploadProgress = const {},
  });

  ImageInputState copyWith({
    List<ImageFile>? selectedImages,
    ImageInputStatus? status,
    String? errorMessage,
    int? maxImages,
    bool? isSelecting,
    Map<String, double>? uploadProgress,
  }) {
    return ImageInputState(
      selectedImages: selectedImages ?? this.selectedImages,
      status: status ?? this.status,
      errorMessage: errorMessage,
      maxImages: maxImages ?? this.maxImages,
      isSelecting: isSelecting ?? this.isSelecting,
      uploadProgress: uploadProgress ?? this.uploadProgress,
    );
  }

  @override
  List<Object?> get props => [
        selectedImages,
        status,
        errorMessage,
        maxImages,
        isSelecting,
        uploadProgress,
      ];

  @override
  String toString() {
    return 'ImageInputState('
        'selectedImages: ${selectedImages.length}, '
        'status: $status, '
        'errorMessage: $errorMessage, '
        'maxImages: $maxImages, '
        'isSelecting: $isSelecting, '
        'uploadProgress: $uploadProgress)';
  }
}
```

### 事件定义
```dart
// lib/features/image_input/bloc/image_input_event.dart
part of 'image_input_cubit.dart';

abstract class ImageInputEvent extends Equatable {
  const ImageInputEvent();

  @override
  List<Object> get props => [];
}

class SelectImagesFromGallery extends ImageInputEvent {}

class CaptureImageFromCamera extends ImageInputEvent {}

class SelectSampleImage extends ImageInputEvent {
  final String sampleImageId;

  const SelectSampleImage(this.sampleImageId);

  @override
  List<Object> get props => [sampleImageId];
}

class SelectFromHistory extends ImageInputEvent {
  final ProcessingHistory historyItem;

  const SelectFromHistory(this.historyItem);

  @override
  List<Object> get props => [historyItem];
}

class AddImage extends ImageInputEvent {
  final ImageFile imageFile;

  const AddImage(this.imageFile);

  @override
  List<Object> get props => [imageFile];
}

class RemoveImage extends ImageInputEvent {
  final String imageId;

  const RemoveImage(this.imageId);

  @override
  List<Object> get props => [imageId];
}

class ClearSelectedImages extends ImageInputEvent {}

class UpdateImageUploadProgress extends ImageInputEvent {
  final String imageId;
  final double progress;

  const UpdateImageUploadProgress(this.imageId, this.progress);

  @override
  List<Object> get props => [imageId, progress];
}

class ValidateImages extends ImageInputEvent {}

class UploadImages extends ImageInputEvent {}

class RetryUpload extends ImageInputEvent {
  final String? imageId;

  const RetryUpload({this.imageId});
}

class CancelUpload extends ImageInputEvent {
  final String imageId;

  const CancelUpload(this.imageId);

  @override
  List<Object> get props => [imageId];
}
```

### Cubit实现
```dart
// lib/features/image_input/bloc/image_input_cubit.dart
class ImageInputCubit extends Cubit<ImageInputState> {
  final ImageRepository _imageRepository;

  ImageInputCubit(this._imageRepository) : super(const ImageInputState());

  // 图片选择相关方法
  Future<void> selectImagesFromGallery() async {
    try {
      emit(state.copyWith(isSelecting: true, status: ImageInputStatus.loading));

      final images = await _imageRepository.pickImagesFromGallery();

      if (images.isEmpty) {
        emit(state.copyWith(
          isSelecting: false,
          status: ImageInputStatus.initial,
        ));
        return;
      }

      final validImages = _validateImages(images);
      await _addValidImages(validImages);

      emit(state.copyWith(
        isSelecting: false,
        status: ImageInputStatus.success,
      ));
    } catch (e) {
      emit(state.copyWith(
        isSelecting: false,
        status: ImageInputStatus.error,
        errorMessage: '选择图片失败: ${e.toString()}',
      ));
    }
  }

  Future<void> captureImageFromCamera() async {
    try {
      emit(state.copyWith(isSelecting: true, status: ImageInputStatus.loading));

      final image = await _imageRepository.captureImageFromCamera();

      if (image == null) {
        emit(state.copyWith(
          isSelecting: false,
          status: ImageInputStatus.initial,
        ));
        return;
      }

      final validImages = _validateImages([image]);
      await _addValidImages(validImages);

      emit(state.copyWith(
        isSelecting: false,
        status: ImageInputStatus.success,
      ));
    } catch (e) {
      emit(state.copyWith(
        isSelecting: false,
        status: ImageInputStatus.error,
        errorMessage: '拍照失败: ${e.toString()}',
      ));
    }
  }

  Future<void> selectSampleImage(String sampleImageId) async {
    try {
      emit(state.copyWith(status: ImageInputStatus.loading));

      final sampleImage = await _imageRepository.getSampleImage(sampleImageId);

      if (sampleImage == null) {
        emit(state.copyWith(
          status: ImageInputStatus.error,
          errorMessage: '样例图片不存在',
        ));
        return;
      }

      await _addValidImages([sampleImage]);

      emit(state.copyWith(status: ImageInputStatus.success));
    } catch (e) {
      emit(state.copyWith(
        status: ImageInputStatus.error,
        errorMessage: '加载样例图片失败: ${e.toString()}',
      ));
    }
  }

  Future<void> selectFromHistory(ProcessingHistory historyItem) async {
    try {
      emit(state.copyWith(status: ImageInputStatus.loading));

      final image = await _imageRepository.getImageFromHistory(historyItem);

      if (image == null) {
        emit(state.copyWith(
          status: ImageInputStatus.error,
          errorMessage: '历史图片不存在',
        ));
        return;
      }

      await _addValidImages([image]);

      emit(state.copyWith(status: ImageInputStatus.success));
    } catch (e) {
      emit(state.copyWith(
        status: ImageInputStatus.error,
        errorMessage: '加载历史图片失败: ${e.toString()}',
      ));
    }
  }

  void removeImage(String imageId) {
    final updatedImages = state.selectedImages
        .where((image) => image.id != imageId)
        .toList();

    final updatedProgress = Map<String, double>.from(state.uploadProgress);
    updatedProgress.remove(imageId);

    emit(state.copyWith(
      selectedImages: updatedImages,
      uploadProgress: updatedProgress,
    ));
  }

  void clearSelectedImages() {
    emit(const ImageInputState());
  }

  void updateImageUploadProgress(String imageId, double progress) {
    final updatedProgress = Map<String, double>.from(state.uploadProgress);
    updatedProgress[imageId] = progress;

    emit(state.copyWith(uploadProgress: updatedProgress));
  }

  // 图片验证逻辑
  List<ImageFile> _validateImages(List<ImageFile> images) {
    final validImages = <ImageFile>[];
    final errors = <String>[];

    for (final image in images) {
      if (!_isValidFormat(image.format)) {
        errors.add('图片格式不支持: ${image.format}');
        continue;
      }

      if (!_isValidSize(image.sizeBytes)) {
        errors.add('图片大小超过限制: ${image.name}');
        continue;
      }

      validImages.add(image);
    }

    if (errors.isNotEmpty && validImages.isNotEmpty) {
      // 可以显示部分错误信息
    }

    return validImages;
  }

  bool _isValidFormat(String format) {
    const supportedFormats = ['JPG', 'JPEG', 'PNG', 'WEBP', 'HEIC'];
    return supportedFormats.contains(format.toUpperCase());
  }

  bool _isValidSize(int sizeBytes) {
    const maxSizeBytes = 20 * 1024 * 1024; // 20MB
    return sizeBytes <= maxSizeBytes;
  }

  Future<void> _addValidImages(List<ImageFile> images) async {
    final currentCount = state.selectedImages.length;
    final availableSlots = state.maxImages - currentCount;

    if (availableSlots <= 0) {
      throw Exception('已达到最大图片数量限制');
    }

    final imagesToAdd = images.take(availableSlots).toList();
    final updatedImages = [...state.selectedImages, ...imagesToAdd];

    emit(state.copyWith(selectedImages: updatedImages));
  }

  // 上传相关逻辑
  Future<void> uploadImages() async {
    if (state.selectedImages.isEmpty) {
      emit(state.copyWith(
        status: ImageInputStatus.error,
        errorMessage: '没有选择图片',
      ));
      return;
    }

    try {
      emit(state.copyWith(status: ImageInputStatus.processing));

      for (final image in state.selectedImages) {
        updateImageUploadProgress(image.id, 0.0);
      }

      // 并行上传图片
      final uploadFutures = state.selectedImages.map((image) =>
        _uploadSingleImage(image)
      ).toList();

      await Future.wait(uploadFutures);

      emit(state.copyWith(
        status: ImageInputStatus.success,
        uploadProgress: {},
      ));
    } catch (e) {
      emit(state.copyWith(
        status: ImageInputStatus.error,
        errorMessage: '上传失败: ${e.toString()}',
      ));
    }
  }

  Future<void> _uploadSingleImage(ImageFile image) async {
    try {
      // 模拟上传进度
      for (int i = 0; i <= 100; i += 10) {
        await Future.delayed(Duration(milliseconds: 100));
        updateImageUploadProgress(image.id, i / 100.0);
      }

      await _imageRepository.uploadImage(image);
    } catch (e) {
      // 单个图片上传失败，但不影响其他图片
      print('Upload failed for ${image.id}: $e');
    }
  }

  void retryUpload([String? imageId]) {
    if (imageId != null) {
      // 重试单个图片
      _uploadSingleImage(
        state.selectedImages.firstWhere((img) => img.id == imageId),
      );
    } else {
      // 重试所有图片
      uploadImages();
    }
  }

  void cancelUpload(String imageId) {
    removeImage(imageId);
  }

  @override
  void onChange(Change<ImageInputState> change) {
    super.onChange(change);
    log('ImageInputState changed: $change');
  }
}
```

---

## 🧠 算法选择状态管理

### 状态定义
```dart
// lib/features/algorithm_select/bloc/algorithm_select_state.dart
part of 'algorithm_select_cubit.dart';

enum AlgorithmSelectStatus {
  initial,           // 初始状态
  loading,           // 加载算法列表
  loaded,            // 算法列表已加载
  recommending,      // 获取推荐算法
  filtering,         // 筛选中
  error,             // 错误状态
}

class AlgorithmSelectState extends Equatable {
  final List<Algorithm> algorithms;
  final List<Algorithm> filteredAlgorithms;
  final List<Algorithm> recommendedAlgorithms;
  final Algorithm? selectedAlgorithm;
  final Set<String> favoriteAlgorithms;
  final AlgorithmSelectStatus status;
  final String? errorMessage;
  final String searchQuery;
  final AlgorithmFilter filter;

  const AlgorithmSelectState({
    this.algorithms = const [],
    this.filteredAlgorithms = const [],
    this.recommendedAlgorithms = const [],
    this.selectedAlgorithm,
    this.favoriteAlgorithms = const {},
    this.status = AlgorithmSelectStatus.initial,
    this.errorMessage,
    this.searchQuery = '',
    this.filter = const AlgorithmFilter(),
  });

  AlgorithmSelectState copyWith({
    List<Algorithm>? algorithms,
    List<Algorithm>? filteredAlgorithms,
    List<Algorithm>? recommendedAlgorithms,
    Algorithm? selectedAlgorithm,
    Set<String>? favoriteAlgorithms,
    AlgorithmSelectStatus? status,
    String? errorMessage,
    String? searchQuery,
    AlgorithmFilter? filter,
  }) {
    return AlgorithmSelectState(
      algorithms: algorithms ?? this.algorithms,
      filteredAlgorithms: filteredAlgorithms ?? this.filteredAlgorithms,
      recommendedAlgorithms: recommendedAlgorithms ?? this.recommendedAlgorithms,
      selectedAlgorithm: selectedAlgorithm ?? this.selectedAlgorithm,
      favoriteAlgorithms: favoriteAlgorithms ?? this.favoriteAlgorithms,
      status: status ?? this.status,
      errorMessage: errorMessage,
      searchQuery: searchQuery ?? this.searchQuery,
      filter: filter ?? this.filter,
    );
  }

  @override
  List<Object?> get props => [
        algorithms,
        filteredAlgorithms,
        recommendedAlgorithms,
        selectedAlgorithm,
        favoriteAlgorithms,
        status,
        errorMessage,
        searchQuery,
        filter,
      ];
}

class AlgorithmFilter extends Equatable {
  final AlgorithmType? type;
  final ProcessingSpeed? speed;
  final QualityLevel? quality;
  final double? minRating;
  final bool favoritesOnly;

  const AlgorithmFilter({
    this.type,
    this.speed,
    this.quality,
    this.minRating,
    this.favoritesOnly = false,
  });

  AlgorithmFilter copyWith({
    AlgorithmType? type,
    ProcessingSpeed? speed,
    QualityLevel? quality,
    double? minRating,
    bool? favoritesOnly,
  }) {
    return AlgorithmFilter(
      type: type ?? this.type,
      speed: speed ?? this.speed,
      quality: quality ?? this.quality,
      minRating: minRating ?? this.minRating,
      favoritesOnly: favoritesOnly ?? this.favoritesOnly,
    );
  }

  @override
  List<Object?> get props => [type, speed, quality, minRating, favoritesOnly];

  bool get isEmpty =>
      type == null &&
      speed == null &&
      quality == null &&
      minRating == null &&
      !favoritesOnly;
}
```

### Cubit实现
```dart
// lib/features/algorithm_select/bloc/algorithm_select_cubit.dart
class AlgorithmSelectCubit extends Cubit<AlgorithmSelectState> {
  final AlgorithmRepository _algorithmRepository;

  AlgorithmSelectCubit(this._algorithmRepository)
      : super(const AlgorithmSelectState());

  Future<void> loadAlgorithms() async {
    try {
      emit(state.copyWith(status: AlgorithmSelectStatus.loading));

      final algorithms = await _algorithmRepository.getAlgorithms();
      final favorites = await _algorithmRepository.getFavoriteAlgorithms();

      emit(state.copyWith(
        algorithms: algorithms,
        filteredAlgorithms: algorithms,
        favoriteAlgorithms: favorites,
        status: AlgorithmSelectStatus.loaded,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: AlgorithmSelectStatus.error,
        errorMessage: '加载算法列表失败: ${e.toString()}',
      ));
    }
  }

  Future<void> getRecommendedAlgorithms(ImageFile imageFile) async {
    try {
      emit(state.copyWith(status: AlgorithmSelectStatus.recommending));

      final recommended = await _algorithmRepository.getRecommendedAlgorithms(imageFile);

      emit(state.copyWith(
        recommendedAlgorithms: recommended,
        status: AlgorithmSelectStatus.loaded,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: AlgorithmSelectStatus.error,
        errorMessage: '获取推荐算法失败: ${e.toString()}',
      ));
    }
  }

  void searchAlgorithms(String query) {
    emit(state.copyWith(searchQuery: query));
    _applyFilter();
  }

  void updateFilter(AlgorithmFilter filter) {
    emit(state.copyWith(filter: filter));
    _applyFilter();
  }

  void clearFilter() {
    emit(state.copyWith(
      filter: const AlgorithmFilter(),
      searchQuery: '',
    ));
    _applyFilter();
  }

  void selectAlgorithm(Algorithm algorithm) {
    emit(state.copyWith(selectedAlgorithm: algorithm));
  }

  void clearSelection() {
    emit(state.copyWith(selectedAlgorithm: null));
  }

  Future<void> toggleFavorite(String algorithmId) async {
    final updatedFavorites = Set<String>.from(state.favoriteAlgorithms);

    if (updatedFavorites.contains(algorithmId)) {
      updatedFavorites.remove(algorithmId);
      await _algorithmRepository.removeFavoriteAlgorithm(algorithmId);
    } else {
      updatedFavorites.add(algorithmId);
      await _algorithmRepository.addFavoriteAlgorithm(algorithmId);
    }

    emit(state.copyWith(favoriteAlgorithms: updatedFavorites));

    // 如果筛选条件包含只显示收藏，重新应用筛选
    if (state.filter.favoritesOnly) {
      _applyFilter();
    }
  }

  void _applyFilter() {
    var filtered = List<Algorithm>.from(state.algorithms);

    // 搜索筛选
    if (state.searchQuery.isNotEmpty) {
      final query = state.searchQuery.toLowerCase();
      filtered = filtered.where((algorithm) {
        return algorithm.name.toLowerCase().contains(query) ||
               algorithm.description.toLowerCase().contains(query) ||
               algorithm.tags.any((tag) => tag.toLowerCase().contains(query));
      }).toList();
    }

    // 类型筛选
    if (state.filter.type != null) {
      filtered = filtered
          .where((algorithm) => algorithm.type == state.filter.type)
          .toList();
    }

    // 速度筛选
    if (state.filter.speed != null) {
      filtered = filtered
          .where((algorithm) => algorithm.speed == state.filter.speed)
          .toList();
    }

    // 质量筛选
    if (state.filter.quality != null) {
      filtered = filtered
          .where((algorithm) => algorithm.quality == state.filter.quality)
          .toList();
    }

    // 评分筛选
    if (state.filter.minRating != null) {
      filtered = filtered
          .where((algorithm) => algorithm.rating >= state.filter.minRating!)
          .toList();
    }

    // 收藏筛选
    if (state.filter.favoritesOnly) {
      filtered = filtered
          .where((algorithm) => state.favoriteAlgorithms.contains(algorithm.id))
          .toList();
    }

    // 排序
    filtered = _sortAlgorithms(filtered);

    emit(state.copyWith(
      filteredAlgorithms: filtered,
      status: AlgorithmSelectStatus.loaded,
    ));
  }

  List<Algorithm> _sortAlgorithms(List<Algorithm> algorithms) {
    // 默认按评分排序，推荐算法优先
    final sorted = List<Algorithm>.from(algorithms);

    sorted.sort((a, b) {
      // 推荐算法排在前面
      final aIsRecommended = state.recommendedAlgorithms.contains(a);
      final bIsRecommended = state.recommendedAlgorithms.contains(b);

      if (aIsRecommended && !bIsRecommended) return -1;
      if (!aIsRecommended && bIsRecommended) return 1;

      // 收藏算法排在前面
      final aIsFavorite = state.favoriteAlgorithms.contains(a.id);
      final bIsFavorite = state.favoriteAlgorithms.contains(b.id);

      if (aIsFavorite && !bIsFavorite) return -1;
      if (!aIsFavorite && bIsFavorite) return 1;

      // 按评分排序
      return b.rating.compareTo(a.rating);
    });

    return sorted;
  }

  Future<void> refreshAlgorithms() async {
    await loadAlgorithms();
  }
}
```

---

## ⚙️ 处理状态管理

### 状态定义
```dart
// lib/features/processing/bloc/processing_state.dart
part of 'processing_cubit.dart';

enum ProcessingStatus {
  initial,       // 初始状态
  validating,    // 验证参数
  queuing,       // 加入队列
  processing,    // 处理中
  paused,        // 已暂停
  completed,     // 处理完成
  error,         // 处理错误
  cancelled,     // 已取消
}

class ProcessingState extends Equatable {
  final List<ImageFile> inputImages;
  final Algorithm selectedAlgorithm;
  final ProcessingParameters parameters;
  final List<ProcessingTask> tasks;
  final ProcessingTask? currentTask;
  final ProcessingStatus status;
  final String? errorMessage;
  final Duration totalEstimatedTime;
  final Duration elapsedTime;
  final bool isBatchProcessing;
  final Map<String, ProcessedImage> results;

  const ProcessingState({
    this.inputImages = const [],
    this.selectedAlgorithm = const Algorithm.empty(),
    this.parameters = const ProcessingParameters(),
    this.tasks = const [],
    this.currentTask,
    this.status = ProcessingStatus.initial,
    this.errorMessage,
    this.totalEstimatedTime = Duration.zero,
    this.elapsedTime = Duration.zero,
    this.isBatchProcessing = false,
    this.results = const {},
  });

  ProcessingState copyWith({
    List<ImageFile>? inputImages,
    Algorithm? selectedAlgorithm,
    ProcessingParameters? parameters,
    List<ProcessingTask>? tasks,
    ProcessingTask? currentTask,
    ProcessingStatus? status,
    String? errorMessage,
    Duration? totalEstimatedTime,
    Duration? elapsedTime,
    bool? isBatchProcessing,
    Map<String, ProcessedImage>? results,
  }) {
    return ProcessingState(
      inputImages: inputImages ?? this.inputImages,
      selectedAlgorithm: selectedAlgorithm ?? this.selectedAlgorithm,
      parameters: parameters ?? this.parameters,
      tasks: tasks ?? this.tasks,
      currentTask: currentTask ?? this.currentTask,
      status: status ?? this.status,
      errorMessage: errorMessage,
      totalEstimatedTime: totalEstimatedTime ?? this.totalEstimatedTime,
      elapsedTime: elapsedTime ?? this.elapsedTime,
      isBatchProcessing: isBatchProcessing ?? this.isBatchProcessing,
      results: results ?? this.results,
    );
  }

  double get overallProgress {
    if (tasks.isEmpty) return 0.0;

    final completedTasks = tasks.where((task) =>
        task.status == TaskStatus.completed).length;
    return completedTasks / tasks.length;
  }

  Duration get remainingTime {
    if (currentTask == null) return Duration.zero;

    final remainingTasks = tasks.where((task) =>
        task.status == TaskStatus.pending ||
        task.status == TaskStatus.processing).length;

    return Duration(
      milliseconds: (remainingTasks *
          selectedAlgorithm.averageProcessingTime.inMilliseconds).round(),
    );
  }

  @override
  List<Object?> get props => [
        inputImages,
        selectedAlgorithm,
        parameters,
        tasks,
        currentTask,
        status,
        errorMessage,
        totalEstimatedTime,
        elapsedTime,
        isBatchProcessing,
        results,
      ];
}
```

### Bloc实现
```dart
// lib/features/processing/bloc/processing_cubit.dart
class ProcessingCubit extends Bloc<ProcessingEvent, ProcessingState> {
  final ProcessingRepository _processingRepository;
  final ImageRepository _imageRepository;
  Timer? _progressTimer;
  StreamSubscription<ProcessingProgress>? _progressSubscription;

  ProcessingCubit(
    this._processingRepository,
    this._imageRepository,
  ) : super(const ProcessingState()) {
    on<StartProcessing>(_onStartProcessing);
    on<PauseProcessing>(_onPauseProcessing);
    on<ResumeProcessing>(_onResumeProcessing);
    on<CancelProcessing>(_onCancelProcessing);
    on<UpdateParameters>(_onUpdateParameters);
    on<UpdateProgress>(_onUpdateProgress);
    on<TaskCompleted>(_onTaskCompleted);
    on<TaskFailed>(_onTaskFailed);
    on<ProcessingCompleted>(_onProcessingCompleted);
  }

  Future<void> _onStartProcessing(
    StartProcessing event,
    Emitter<ProcessingState> emit,
  ) async {
    try {
      emit(state.copyWith(
        status: ProcessingStatus.validating,
        errorMessage: null,
      ));

      // 验证输入参数
      await _validateInputs(event.images, event.algorithm, event.parameters);

      emit(state.copyWith(
        inputImages: event.images,
        selectedAlgorithm: event.algorithm,
        parameters: event.parameters,
        isBatchProcessing: event.images.length > 1,
        status: ProcessingStatus.queuing,
      ));

      // 创建处理任务
      final tasks = await _createTasks(event.images, event.algorithm, event.parameters);

      // 计算预估时间
      final totalEstimatedTime = _calculateEstimatedTime(tasks, event.algorithm);

      emit(state.copyWith(
        tasks: tasks,
        totalEstimatedTime: totalEstimatedTime,
        status: ProcessingStatus.processing,
        elapsedTime: Duration.zero,
      ));

      // 开始处理
      await _startProcessingStream(tasks);

    } catch (e) {
      emit(state.copyWith(
        status: ProcessingStatus.error,
        errorMessage: e.toString(),
      ));
    }
  }

  Future<void> _onPauseProcessing(
    PauseProcessing event,
    Emitter<ProcessingState> emit,
  ) async {
    if (state.currentTask != null) {
      await _processingRepository.pauseTask(state.currentTask!.id);

      emit(state.copyWith(
        status: ProcessingStatus.paused,
      ));

      _progressTimer?.cancel();
    }
  }

  Future<void> _onResumeProcessing(
    ResumeProcessing event,
    Emitter<ProcessingState> emit,
  ) async {
    if (state.currentTask != null) {
      await _processingRepository.resumeTask(state.currentTask!.id);

      emit(state.copyWith(
        status: ProcessingStatus.processing,
      ));

      _startProgressTimer();
    }
  }

  Future<void> _onCancelProcessing(
    CancelProcessing event,
    Emitter<ProcessingState> emit,
  ) async {
    // 取消当前任务
    if (state.currentTask != null) {
      await _processingRepository.cancelTask(state.currentTask!.id);
    }

    // 取消所有待处理任务
    for (final task in state.tasks) {
      if (task.status == TaskStatus.pending) {
        await _processingRepository.cancelTask(task.id);
      }
    }

    _progressTimer?.cancel();
    _progressSubscription?.cancel();

    emit(state.copyWith(
      status: ProcessingStatus.cancelled,
      currentTask: null,
    ));
  }

  Future<void> _onUpdateParameters(
    UpdateParameters event,
    Emitter<ProcessingState> emit,
  ) async {
    emit(state.copyWith(
      parameters: event.parameters,
    ));

    // 如果当前有任务在处理，需要重新开始
    if (state.status == ProcessingStatus.processing) {
      add(CancelProcessing());
      add(StartProcessing(
        images: state.inputImages,
        algorithm: state.selectedAlgorithm,
        parameters: event.parameters,
      ));
    }
  }

  Future<void> _onUpdateProgress(
    UpdateProgress event,
    Emitter<ProcessingState> emit,
  ) async {
    // 更新当前任务进度
    final updatedTasks = state.tasks.map((task) {
      if (task.id == event.taskId) {
        return task.copyWith(
          progress: event.progress,
          status: event.status,
          estimatedRemainingTime: event.estimatedRemainingTime,
        );
      }
      return task;
    }).toList();

    final currentTask = updatedTasks.firstWhere(
      (task) => task.id == event.taskId,
    );

    emit(state.copyWith(
      tasks: updatedTasks,
      currentTask: currentTask,
      elapsedTime: event.elapsedTime,
    ));
  }

  Future<void> _onTaskCompleted(
    TaskCompleted event,
    Emitter<ProcessingState> emit,
  ) async {
    final updatedResults = Map<String, ProcessedImage>.from(state.results);
    updatedResults[event.taskId] = event.result;

    final updatedTasks = state.tasks.map((task) {
      if (task.id == event.taskId) {
        return task.copyWith(
          status: TaskStatus.completed,
          progress: 1.0,
          result: event.result,
        );
      }
      return task;
    }).toList();

    emit(state.copyWith(
      results: updatedResults,
      tasks: updatedTasks,
    ));

    // 检查是否所有任务都完成
    if (_areAllTasksCompleted(updatedTasks)) {
      add(ProcessingCompleted());
    } else {
      // 开始下一个任务
      _startNextTask(updatedTasks);
    }
  }

  Future<void> _onTaskFailed(
    TaskFailed event,
    Emitter<ProcessingState> emit,
  ) async {
    final updatedTasks = state.tasks.map((task) {
      if (task.id == event.taskId) {
        return task.copyWith(
          status: TaskStatus.failed,
          errorMessage: event.error,
        );
      }
      return task;
    }).toList();

    emit(state.copyWith(
      tasks: updatedTasks,
      status: ProcessingStatus.error,
      errorMessage: '任务失败: ${event.error}',
    ));
  }

  Future<void> _onProcessingCompleted(
    ProcessingCompleted event,
    Emitter<ProcessingState> emit,
  ) async {
    _progressTimer?.cancel();
    _progressSubscription?.cancel();

    emit(state.copyWith(
      status: ProcessingStatus.completed,
      currentTask: null,
    ));

    // 保存处理历史
    await _saveProcessingHistory();
  }

  // 私有辅助方法
  Future<void> _validateInputs(
    List<ImageFile> images,
    Algorithm algorithm,
    ProcessingParameters parameters,
  ) async {
    if (images.isEmpty) {
      throw Exception('没有选择图片');
    }

    if (algorithm.id.isEmpty) {
      throw Exception('没有选择算法');
    }

    for (final image in images) {
      if (!await _imageRepository.validateImage(image)) {
        throw Exception('图片验证失败: ${image.name}');
      }
    }
  }

  Future<List<ProcessingTask>> _createTasks(
    List<ImageFile> images,
    Algorithm algorithm,
    ProcessingParameters parameters,
  ) async {
    return images.map((image) => ProcessingTask(
      id: uuid.v4(),
      imageFile: image,
      algorithm: algorithm,
      parameters: parameters,
      status: TaskStatus.pending,
      progress: 0.0,
      createdAt: DateTime.now(),
    )).toList();
  }

  Duration _calculateEstimatedTime(
    List<ProcessingTask> tasks,
    Algorithm algorithm,
  ) {
    return Duration(
      milliseconds: tasks.length * algorithm.averageProcessingTime.inMilliseconds,
    );
  }

  Future<void> _startProcessingStream(List<ProcessingTask> tasks) async {
    _progressSubscription = _processingRepository
        .getProcessingStream(tasks)
        .listen((progress) {
          if (progress is ProcessingProgressUpdate) {
            add(UpdateProgress(
              taskId: progress.taskId,
              progress: progress.progress,
              status: progress.status,
              estimatedRemainingTime: progress.estimatedRemainingTime,
              elapsedTime: progress.elapsedTime,
            ));
          } else if (progress is TaskCompletion) {
            add(TaskCompleted(
              taskId: progress.taskId,
              result: progress.result,
            ));
          } else if (progress is TaskFailure) {
            add(TaskFailed(
              taskId: progress.taskId,
              error: progress.error,
            ));
          }
        });

    // 开始第一个任务
    _startNextTask(tasks);
    _startProgressTimer();
  }

  void _startNextTask(List<ProcessingTask> tasks) {
    final pendingTask = tasks.firstWhere(
      (task) => task.status == TaskStatus.pending,
      orElse: () => throw StateError('No pending tasks found'),
    );

    _processingRepository.startTask(pendingTask);
  }

  void _startProgressTimer() {
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(Duration(milliseconds: 100), (timer) {
      // 更新经过时间
      final updatedElapsedTime = state.elapsedTime + Duration(milliseconds: 100);
      // 可以在这里添加更多的进度更新逻辑
    });
  }

  bool _areAllTasksCompleted(List<ProcessingTask> tasks) {
    return tasks.every((task) =>
        task.status == TaskStatus.completed ||
        task.status == TaskStatus.failed);
  }

  Future<void> _saveProcessingHistory() async {
    try {
      final history = ProcessingHistory(
        id: uuid.v4(),
        images: state.inputImages,
        algorithm: state.selectedAlgorithm,
        parameters: state.parameters,
        results: state.results.values.toList(),
        processingTime: state.elapsedTime,
        createdAt: DateTime.now(),
      );

      await _imageRepository.saveProcessingHistory(history);
    } catch (e) {
      log('Failed to save processing history: $e');
    }
  }

  @override
  Future<void> close() {
    _progressTimer?.cancel();
    _progressSubscription?.cancel();
    return super.close();
  }
}
```

---

## 📊 效果对比状态管理

### 状态定义
```dart
// lib/features/comparison/bloc/comparison_state.dart
part of 'comparison_cubit.dart';

enum ComparisonMode {
  sideBySide,    // 并排对比
  overlay,       // 重叠对比
  slider,        // 滑动对比
  magnifier,     // 放大镜
  filter,        // 滤镜调节
  metrics,       // 指标评估
}

class ComparisonState extends Equatable {
  final ImageData originalImage;
  final ImageData processedImage;
  final ComparisonMode currentMode;
  final Map<ComparisonMode, ComparisonSettings> modeSettings;
  final List<ImageQualityMetric> metrics;
  final bool isLoading;
  final String? errorMessage;
  final List<ComparisonHistory> history;

  const ComparisonState({
    required this.originalImage,
    required this.processedImage,
    this.currentMode = ComparisonMode.sideBySide,
    this.modeSettings = const {},
    this.metrics = const [],
    this.isLoading = false,
    this.errorMessage,
    this.history = const [],
  });

  ComparisonState copyWith({
    ImageData? originalImage,
    ImageData? processedImage,
    ComparisonMode? currentMode,
    Map<ComparisonMode, ComparisonSettings>? modeSettings,
    List<ImageQualityMetric>? metrics,
    bool? isLoading,
    String? errorMessage,
    List<ComparisonHistory>? history,
  }) {
    return ComparisonState(
      originalImage: originalImage ?? this.originalImage,
      processedImage: processedImage ?? this.processedImage,
      currentMode: currentMode ?? this.currentMode,
      modeSettings: modeSettings ?? this.modeSettings,
      metrics: metrics ?? this.metrics,
      isLoading: isLoading ?? this.isLoading,
      errorMessage: errorMessage,
      history: history ?? this.history,
    );
  }

  ComparisonSettings? get currentSettings =>
      modeSettings[currentMode] ?? _getDefaultSettings(currentMode);

  ComparisonSettings _getDefaultSettings(ComparisonMode mode) {
    switch (mode) {
      case ComparisonMode.sideBySide:
        return const ComparisonSettings(
          splitPosition: 0.5,
          splitDirection: Axis.horizontal,
        );
      case ComparisonMode.overlay:
        return const ComparisonSettings(
          overlayOpacity: 0.5,
          originalOnTop: true,
        );
      case ComparisonMode.slider:
        return const ComparisonSettings(
          sliderPosition: 0.5,
          sliderDirection: Axis.horizontal,
        );
      case ComparisonMode.magnifier:
        return const ComparisonSettings(
          magnifierSize: 150,
          magnification: 2.0,
          showOriginalInMagnifier: true,
        );
      case ComparisonMode.filter:
        return const ComparisonSettings(
          brightness: 0.0,
          contrast: 0.0,
          saturation: 0.0,
        );
      case ComparisonMode.metrics:
        return const ComparisonSettings();
    }
  }

  @override
  List<Object?> get props => [
        originalImage,
        processedImage,
        currentMode,
        modeSettings,
        metrics,
        isLoading,
        errorMessage,
        history,
      ];
}

class ComparisonSettings extends Equatable {
  final double splitPosition;
  final Axis splitDirection;
  final double overlayOpacity;
  final bool originalOnTop;
  final double sliderPosition;
  final Axis sliderDirection;
  final double magnifierSize;
  final double magnification;
  final bool showOriginalInMagnifier;
  final double brightness;
  final double contrast;
  final double saturation;

  const ComparisonSettings({
    this.splitPosition = 0.5,
    this.splitDirection = Axis.horizontal,
    this.overlayOpacity = 0.5,
    this.originalOnTop = true,
    this.sliderPosition = 0.5,
    this.sliderDirection = Axis.horizontal,
    this.magnifierSize = 150,
    this.magnification = 2.0,
    this.showOriginalInMagnifier = true,
    this.brightness = 0.0,
    this.contrast = 0.0,
    this.saturation = 0.0,
  });

  ComparisonSettings copyWith({
    double? splitPosition,
    Axis? splitDirection,
    double? overlayOpacity,
    bool? originalOnTop,
    double? sliderPosition,
    Axis? sliderDirection,
    double? magnifierSize,
    double? magnification,
    bool? showOriginalInMagnifier,
    double? brightness,
    double? contrast,
    double? saturation,
  }) {
    return ComparisonSettings(
      splitPosition: splitPosition ?? this.splitPosition,
      splitDirection: splitDirection ?? this.splitDirection,
      overlayOpacity: overlayOpacity ?? this.overlayOpacity,
      originalOnTop: originalOnTop ?? this.originalOnTop,
      sliderPosition: sliderPosition ?? this.sliderPosition,
      sliderDirection: sliderDirection ?? this.sliderDirection,
      magnifierSize: magnifierSize ?? this.magnifierSize,
      magnification: magnification ?? this.magnification,
      showOriginalInMagnifier:
          showOriginalInMagnifier ?? this.showOriginalInMagnifier,
      brightness: brightness ?? this.brightness,
      contrast: contrast ?? this.contrast,
      saturation: saturation ?? this.saturation,
    );
  }

  @override
  List<Object?> get props => [
        splitPosition,
        splitDirection,
        overlayOpacity,
        originalOnTop,
        sliderPosition,
        sliderDirection,
        magnifierSize,
        magnification,
        showOriginalInMagnifier,
        brightness,
        contrast,
        saturation,
      ];
}
```

---

## 🔁 状态持久化

### HydratedBloc配置
```dart
// lib/core/hydration/hydrated_bloc_config.dart
class AppHydratedBlocConfig {
  static void configure() {
    // 配置存储
    HydratedBloc.storage = HydratedStorage(
      storage: SharedPreferencesStorage(),
    );

    // 自定义JSON转换器
    HydratedBloc.transformers = [
      ImageInputHydratedTransformer(),
      AlgorithmSelectHydratedTransformer(),
      UserPreferencesHydratedTransformer(),
    ];
  }
}

// 图像输入状态持久化
class ImageInputHydratedTransformer extends HydratedTransformer<ImageInputState> {
  @override
  ImageInputState fromJson(Map<String, dynamic> json) {
    return ImageInputState(
      selectedImages: (json['selectedImages'] as List?)
          ?.map((e) => ImageFile.fromJson(e))
          .toList() ?? [],
      maxImages: json['maxImages'] as int? ?? 5,
      favoriteAlgorithms: (json['favoriteAlgorithms'] as List?)
          ?.map((e) => e.toString())
          .toSet() ?? {},
    );
  }

  @override
  Map<String, dynamic> toJson(ImageInputState state) {
    return {
      'selectedImages': state.selectedImages.map((e) => e.toJson()).toList(),
      'maxImages': state.maxImages,
      'favoriteAlgorithms': state.favoriteAlgorithms.toList(),
    };
  }
}
```

---

## 📊 性能优化策略

### 状态缓存
```dart
// lib/core/cache/state_cache.dart
class StateCache {
  static final Map<String, dynamic> _cache = {};
  static const Duration _defaultExpiry = Duration(minutes: 5);

  static T? get<T>(String key) {
    final cached = _cache[key];
    if (cached == null) return null;

    final cachedItem = cached as _CachedItem<T>;
    if (DateTime.now().isAfter(cachedItem.expiry)) {
      _cache.remove(key);
      return null;
    }

    return cachedItem.value;
  }

  static void set<T>(String key, T value, [Duration? expiry]) {
    _cache[key] = _CachedItem(
      value: value,
      expiry: DateTime.now().add(expiry ?? _defaultExpiry),
    );
  }

  static void clear() {
    _cache.clear();
  }

  static void remove(String key) {
    _cache.remove(key);
  }
}

class _CachedItem<T> {
  final T value;
  final DateTime expiry;

  _CachedItem({
    required this.value,
    required this.expiry,
  });
}
```

### 防抖处理
```dart
// lib/utils/debouncer.dart
class Debouncer {
  final Duration delay;
  Timer? _timer;
  VoidCallback? _callback;

  Debouncer({required this.delay});

  void run(VoidCallback callback) {
    _callback = callback;
    _timer?.cancel();
    _timer = Timer(delay, _execute);
  }

  void _execute() {
    _callback?.call();
  }

  void cancel() {
    _timer?.cancel();
  }

  void dispose() {
    _timer?.cancel();
    _callback = null;
  }
}

// 使用示例
class SearchCubit extends Cubit<SearchState> {
  final Debouncer _debouncer = Debouncer(delay: Duration(milliseconds: 300));

  void updateSearchQuery(String query) {
    _debouncer.run(() {
      // 执行搜索逻辑
      _performSearch(query);
    });
  }

  @override
  Future<void> close() {
    _debouncer.dispose();
    return super.close();
  }
}
```

---

## 🧪 测试策略

### 状态管理测试
```dart
// test/features/algorithm_select/bloc/algorithm_select_cubit_test.dart
void main() {
  group('AlgorithmSelectCubit', () {
    late AlgorithmSelectCubit cubit;
    late MockAlgorithmRepository mockRepository;

    setUp(() {
      mockRepository = MockAlgorithmRepository();
      cubit = AlgorithmSelectCubit(mockRepository);
    });

    tearDown(() {
      cubit.close();
    });

    test('初始状态正确', () {
      expect(cubit.state.status, AlgorithmSelectStatus.initial);
      expect(cubit.state.algorithms, isEmpty);
      expect(cubit.state.filteredAlgorithms, isEmpty);
    });

    test('加载算法列表成功', () async {
      // Arrange
      final algorithms = [
        Algorithm.test('1', 'DCP', 'Dark Channel Prior'),
        Algorithm.test('2', 'AOD-Net', 'All-in-One Dehazing Network'),
      ];

      when(() => mockRepository.getAlgorithms())
          .thenAnswer((_) async => algorithms);
      when(() => mockRepository.getFavoriteAlgorithms())
          .thenAnswer((_) async => <String>{'1'});

      // Act
      await cubit.loadAlgorithms();

      // Assert
      expect(cubit.state.status, AlgorithmSelectStatus.loaded);
      expect(cubit.state.algorithms, algorithms);
      expect(cubit.state.filteredAlgorithms, algorithms);
      expect(cubit.state.favoriteAlgorithms, contains('1'));

      verify(() => mockRepository.getAlgorithms()).called(1);
      verify(() => mockRepository.getFavoriteAlgorithms()).called(1);
    });

    test('搜索算法正确过滤', () {
      // Arrange
      cubit.emit(cubit.state.copyWith(
        algorithms: [
          Algorithm.test('1', 'DCP', 'Dark Channel Prior'),
          Algorithm.test('2', 'AOD-Net', 'All-in-One Dehazing Network'),
        ],
        filteredAlgorithms: [
          Algorithm.test('1', 'DCP', 'Dark Channel Prior'),
          Algorithm.test('2', 'AOD-Net', 'All-in-One Dehazing Network'),
        ],
      ));

      // Act
      cubit.searchAlgorithms('DCP');

      // Assert
      expect(cubit.state.searchQuery, 'DCP');
      expect(cubit.state.filteredAlgorithms, hasLength(1));
      expect(cubit.state.filteredAlgorithms.first.name, 'DCP');
    });

    test('切换收藏状态', () async {
      // Arrange
      const algorithmId = '1';
      cubit.emit(cubit.state.copyWith(
        algorithms: [Algorithm.test(algorithmId, 'DCP', 'Dark Channel Prior')],
        favoriteAlgorithms: <String>{},
      ));

      when(() => mockRepository.addFavoriteAlgorithm(algorithmId))
          .thenAnswer((_) async {});

      // Act
      await cubit.toggleFavorite(algorithmId);

      // Assert
      expect(cubit.state.favoriteAlgorithms, contains(algorithmId));
      verify(() => mockRepository.addFavoriteAlgorithm(algorithmId)).called(1);
    });
  });
}
```

---

**文档版本**: v1.0
**最后更新**: 2025-11-22
**参考文档**: [架构设计](../design/02-architecture.md)、[UI组件设计](02-ui-components.md)