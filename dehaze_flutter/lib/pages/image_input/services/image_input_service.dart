import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:image/image.dart' as img;

import '../../../core/constants/api_constants.dart';
import '../models/image_input_model.dart';

/// 图片输入服务
///
/// 处理图片验证、压缩、下载、样例获取等功能
class ImageInputService {
  const ImageInputService(this._dio);

  final Dio _dio;

  // 文件限制常量
  static const int maxFileSizeBytes = 20 * 1024 * 1024; // 20MB
  static const int compressionThresholdBytes = 5 * 1024 * 1024; // 5MB
  static const int maxWidth = 8000;
  static const int maxHeight = 8000;
  static const int compressionQuality = 85;

  static const List<String> supportedFormats = [
    'jpg',
    'jpeg',
    'png',
    'webp',
    'heic',
  ];

  /// 验证图片格式和大小（基于字节流，跨平台）
  Future<ImageValidationResult> validateImage(
    Uint8List bytes,
    String filename,
  ) async {
    // 检查文件大小
    if (bytes.isEmpty) {
      return const ImageValidationResult(
        isValid: false,
        errorMessage: '文件为空',
      );
    }
    if (bytes.length > maxFileSizeBytes) {
      return const ImageValidationResult(
        isValid: false,
        errorMessage: '图片大小超过20MB，请选择较小的图片',
      );
    }

    // 检查文件格式
    final extension = filename.split('.').last.toLowerCase();
    if (!supportedFormats.contains(extension)) {
      return const ImageValidationResult(
        isValid: false,
        errorMessage: '不支持该图片格式，请选择JPG/PNG/WEBP/HEIC格式',
      );
    }

    // 检查图片尺寸
    try {
      final image = img.decodeImage(bytes);
      if (image == null) {
        return const ImageValidationResult(
          isValid: false,
          errorMessage: '无法解析图片，请选择有效的图片文件',
        );
      }

      if (image.width > maxWidth || image.height > maxHeight) {
        return const ImageValidationResult(
          isValid: false,
          errorMessage: '图片分辨率超过8000×8000，请选择较小的图片',
        );
      }

      // 检查是否需要压缩
      final needsCompression = bytes.length > compressionThresholdBytes;

      return ImageValidationResult(
        isValid: true,
        needsCompression: needsCompression,
      );
    } catch (e) {
      return ImageValidationResult(
        isValid: false,
        errorMessage: '图片解析失败: $e',
      );
    }
  }

  /// 压缩图片（>5MB 自动压缩，返回压缩后的字节流）
  Future<Uint8List> compressImage(
    Uint8List bytes, {
    int quality = compressionQuality,
  }) async {
    try {
      final image = img.decodeImage(bytes);
      if (image == null) {
        throw Exception('无法解析图片');
      }

      // 编码为 JPEG 并压缩
      return Uint8List.fromList(img.encodeJpg(image, quality: quality));
    } catch (e) {
      throw Exception('图片压缩失败: $e');
    }
  }

  /// 获取图片信息（宽高、大小，基于字节流）
  Future<ImageInfo> getImageInfo(Uint8List bytes) async {
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('无法解析图片');
    }

    return ImageInfo(
      width: image.width,
      height: image.height,
      fileSize: bytes.length,
    );
  }

  /// 从 URL 下载图片为字节流（跨平台）
  Future<Uint8List> downloadImageBytes(String url) async {
    try {
      final response = await _dio.get<List<int>>(
        url,
        options: Options(responseType: ResponseType.bytes),
      );
      return Uint8List.fromList(response.data ?? []);
    } catch (e) {
      throw Exception('图片下载失败: $e');
    }
  }

  /// 获取样例图片列表
  ///
  /// 样例库复用数据集的公开数据项：
  /// GET /dataset-items 返回数据项及其有雾图，取有雾图作为样例。
  Future<List<SampleImageModel>> fetchSamples({
    SampleCategory? category,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.datasetItems,
      queryParameters: const {'pageNum': 1, 'pageSize': 50},
    );

    // 业务状态码由 ResponseInterceptor 统一拦截，此处响应均为成功，直接读取 data
    final data = response.data!['data'] as Map<String, dynamic>?;
    final items = (data?['list'] as List<dynamic>? ?? [])
        .whereType<Map<String, dynamic>>();

    final samples = <SampleImageModel>[];
    for (final item in items) {
      // 获取清晰图（GT）URL，用于后续指标评估
      final clearImage = item['clearImage'] as Map<String, dynamic>?;
      final cleanUrl = clearImage?['url'] as String?;

      final hazyImages = (item['hazyImages'] as List<dynamic>? ?? [])
          .whereType<Map<String, dynamic>>();
      for (final image in hazyImages) {
        final url = image['url'] as String?;
        if (url == null || url.isEmpty) continue;

        samples.add(SampleImageModel(
          id: image['id'] as int? ?? 0,
          name: item['name'] as String? ??
              image['fileName'] as String? ??
              '样例图片',
          url: url,
          category: _categoryFromHazeLevel(image['hazeLevel'] as String?),
          difficulty: _difficultyFromHazeLevel(image['hazeLevel'] as String?),
          sceneType:
              image['sceneType'] as String? ?? item['sceneType'] as String?,
          cleanUrl: cleanUrl,
        ));
      }
    }

    // 按分类筛选（all 返回全部）
    if (category == null || category == SampleCategory.all) {
      return samples;
    }
    return samples.where((s) => s.category == category).toList();
  }

  /// 雾霾程度 → 样例分类
  SampleCategory _categoryFromHazeLevel(String? hazeLevel) {
    switch (hazeLevel) {
      case 'light':
        return SampleCategory.light;
      case 'medium':
        return SampleCategory.medium;
      case 'heavy':
        return SampleCategory.heavy;
      default:
        return SampleCategory.special;
    }
  }

  /// 雾霾程度 → 难度等级
  DifficultyLevel _difficultyFromHazeLevel(String? hazeLevel) {
    switch (hazeLevel) {
      case 'light':
        return DifficultyLevel.easy;
      case 'medium':
        return DifficultyLevel.medium;
      default:
        return DifficultyLevel.hard;
    }
  }
}

/// 图片信息
class ImageInfo {
  const ImageInfo({
    required this.width,
    required this.height,
    required this.fileSize,
  });

  final int width;
  final int height;
  final int fileSize;
}
