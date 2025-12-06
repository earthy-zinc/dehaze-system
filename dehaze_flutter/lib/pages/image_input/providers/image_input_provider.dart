import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart' as picker;

import '../../../providers/providers.dart';
import '../models/image_input_model.dart';
import '../services/image_input_service.dart';

// ==================== 服务 Provider ====================

/// 图片输入服务 Provider
final imageInputServiceProvider = Provider<ImageInputService>((ref) {
  final dio = ref.watch<Dio>(dioClientProvider);
  return ImageInputService(dio);
});

// ==================== 状态 Provider ====================

/// 当前输入方式
final inputMethodProvider = StateProvider<InputMethod>((ref) => InputMethod.upload);

/// 选中的图片
final selectedImageProvider = StateProvider<SelectedImageModel?>((ref) => null);

/// 上传进度状态
final uploadProgressProvider = StateProvider<UploadProgress>((ref) => UploadProgress.idle);

// ==================== 图片输入 Notifier ====================

/// 图片输入状态管理
class ImageInputNotifier extends StateNotifier<AsyncValue<SelectedImageModel?>> {
  ImageInputNotifier(this._service, this._ref) : super(const AsyncValue.data(null));

  final ImageInputService _service;
  final Ref _ref;

  final _picker = picker.ImagePicker();

  /// 从相册选择图片
  Future<void> pickImage() async {
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 0,
      status: UploadStatus.selecting,
    );

    try {
      final pickedFile = await _picker.pickImage(
        source: picker.ImageSource.gallery,
        maxWidth: 4096,
        maxHeight: 4096,
      );

      if (pickedFile == null) {
        _ref.read(uploadProgressProvider.notifier).state = UploadProgress.idle;
        return;
      }

      await _processFile(File(pickedFile.path), ImageSource.upload);
    } catch (e) {
      _ref.read(uploadProgressProvider.notifier).state = UploadProgress(
        progress: 0,
        status: UploadStatus.error,
        errorMessage: '选择图片失败: $e',
      );
      state = AsyncValue.error(e, StackTrace.current);
    }
  }

  /// 拍照
  Future<void> captureImage() async {
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 0,
      status: UploadStatus.selecting,
    );

    try {
      final pickedFile = await _picker.pickImage(
        source: picker.ImageSource.camera,
        maxWidth: 4096,
        maxHeight: 4096,
        preferredCameraDevice: picker.CameraDevice.rear,
      );

      if (pickedFile == null) {
        _ref.read(uploadProgressProvider.notifier).state = UploadProgress.idle;
        return;
      }

      await _processFile(File(pickedFile.path), ImageSource.camera);
    } catch (e) {
      _ref.read(uploadProgressProvider.notifier).state = UploadProgress(
        progress: 0,
        status: UploadStatus.error,
        errorMessage: '拍照失败: $e',
      );
      state = AsyncValue.error(e, StackTrace.current);
    }
  }

  /// 处理文件（验证、压缩）
  Future<void> _processFile(File file, ImageSource source) async {
    // 验证图片
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 0.2,
      status: UploadStatus.validating,
    );

    final validation = await _service.validateImage(file);
    if (!validation.isValid) {
      _ref.read(uploadProgressProvider.notifier).state = UploadProgress(
        progress: 0,
        status: UploadStatus.error,
        errorMessage: validation.errorMessage,
      );
      return;
    }

    // 压缩图片（如果需要）
    File processedFile = file;
    if (validation.needsCompression) {
      _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
        progress: 0.4,
        status: UploadStatus.compressing,
      );

      processedFile = await _service.compressImage(file);
    }

    // 获取图片信息
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 0.8,
      status: UploadStatus.uploading,
    );

    final imageInfo = await _service.getImageInfo(processedFile);

    // 创建选中的图片模型
    final selectedImage = SelectedImageModel(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      url: processedFile.path,
      localPath: processedFile.path,
      filename: file.path.split('/').last,
      width: imageInfo.width,
      height: imageInfo.height,
      fileSize: imageInfo.fileSize,
      source: source,
    );

    // 更新状态
    _ref.read(selectedImageProvider.notifier).state = selectedImage;
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 1.0,
      status: UploadStatus.success,
    );
    state = AsyncValue.data(selectedImage);
  }

  /// 选择样例图片
  Future<void> selectSampleImage(SampleImageModel sample) async {
    state = const AsyncValue.loading();
    _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
      progress: 0.3,
      status: UploadStatus.uploading,
    );

    try {
      // 下载样例图片
      final file = await _service.downloadImage(sample.url);
      final imageInfo = await _service.getImageInfo(file);

      final selectedImage = SelectedImageModel(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        url: sample.url,
        localPath: file.path,
        filename: '${sample.name}.jpg',
        width: imageInfo.width,
        height: imageInfo.height,
        fileSize: imageInfo.fileSize,
        source: ImageSource.sample,
        sampleInfo: sample,
      );

      _ref.read(selectedImageProvider.notifier).state = selectedImage;
      _ref.read(uploadProgressProvider.notifier).state = const UploadProgress(
        progress: 1.0,
        status: UploadStatus.success,
      );
      state = AsyncValue.data(selectedImage);
    } catch (e) {
      _ref.read(uploadProgressProvider.notifier).state = UploadProgress(
        progress: 0,
        status: UploadStatus.error,
        errorMessage: '加载样例图片失败: $e',
      );
      state = AsyncValue.error(e, StackTrace.current);
    }
  }

  /// 从历史记录加载
  Future<void> loadFromHistory(HistoryRecordModel record) async {
    state = const AsyncValue.loading();

    try {
      // 从历史记录创建选中的图片
      final selectedImage = SelectedImageModel(
        id: record.id,
        url: record.originalThumbnail,
        filename: record.filename,
        width: 0, // 历史记录可能没有保存尺寸信息
        height: 0,
        fileSize: 0,
        source: ImageSource.history,
      );

      _ref.read(selectedImageProvider.notifier).state = selectedImage;
      state = AsyncValue.data(selectedImage);
    } catch (e) {
      state = AsyncValue.error(e, StackTrace.current);
    }
  }

  /// 清除选中的图片
  void clearSelection() {
    _ref.read(selectedImageProvider.notifier).state = null;
    _ref.read(uploadProgressProvider.notifier).state = UploadProgress.idle;
    state = const AsyncValue.data(null);
  }
}

/// 图片输入 Provider
final imageInputProvider =
    StateNotifierProvider<ImageInputNotifier, AsyncValue<SelectedImageModel?>>((ref) {
  final service = ref.watch(imageInputServiceProvider);
  return ImageInputNotifier(service, ref);
});
