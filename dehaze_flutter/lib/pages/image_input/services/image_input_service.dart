import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:image/image.dart' as img;

import '../models/image_input_model.dart';

/// 图片输入服务
///
/// 处理图片验证、压缩、上传等功能
class ImageInputService {
  const ImageInputService(this._dio);

  final Dio _dio;

  // 文件限制常量
  static const int maxFileSizeBytes = 20 * 1024 * 1024; // 20MB
  static const int compressionThresholdBytes = 5 * 1024 * 1024; // 5MB
  static const int maxWidth = 8000;
  static const int maxHeight = 8000;
  static const int minWidth = 640;
  static const int minHeight = 480;
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
  Future<List<SampleImageModel>> fetchSamples({
    SampleCategory? category,
  }) async {
    try {
      final response = await _dio.get<Map<String, dynamic>>(
        '/samples',
        queryParameters: {
          if (category != null && category != SampleCategory.all)
            'category': category.name,
        },
      );

      if (response.statusCode == 200 && response.data?['code']?.toString() == '00000') {
        final data = response.data!['data'] as List<dynamic>;
        return data
            .map((e) => SampleImageModel.fromJson(e as Map<String, dynamic>))
            .toList();
      }
      throw Exception('获取样例图片失败');
    } on DioException catch (e) {
      // Mock 数据用于开发阶段
      if (e.type == DioExceptionType.connectionError ||
          e.type == DioExceptionType.connectionTimeout) {
        return _getMockSamples(category);
      }
      rethrow;
    }
  }

  /// Mock 样例数据
  List<SampleImageModel> _getMockSamples(SampleCategory? category) {
    final allSamples = [
      // 轻度雾霾
      const SampleImageModel(
        id: 1,
        name: '轻度雾霾-城市街道',
        url: 'https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800',
        category: SampleCategory.light,
        difficulty: DifficultyLevel.easy,
        sceneType: '城市',
      ),
      const SampleImageModel(
        id: 2,
        name: '轻度雾霾-公园景观',
        url: 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800',
        category: SampleCategory.light,
        difficulty: DifficultyLevel.easy,
        sceneType: '风景',
      ),
      const SampleImageModel(
        id: 3,
        name: '轻度雾霾-建筑物',
        url: 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800',
        category: SampleCategory.light,
        difficulty: DifficultyLevel.easy,
        sceneType: '建筑',
      ),
      const SampleImageModel(
        id: 4,
        name: '轻度雾霾-山景',
        url: 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800',
        category: SampleCategory.light,
        difficulty: DifficultyLevel.easy,
        sceneType: '山景',
      ),
      const SampleImageModel(
        id: 5,
        name: '轻度雾霾-湖泊',
        url: 'https://images.unsplash.com/photo-1439066615861-d1af74d74000?w=800',
        category: SampleCategory.light,
        difficulty: DifficultyLevel.easy,
        sceneType: '湖泊',
      ),
      // 中度雾霾
      const SampleImageModel(
        id: 6,
        name: '中度雾霾-城市天际线',
        url: 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800',
        category: SampleCategory.medium,
        difficulty: DifficultyLevel.medium,
        sceneType: '城市',
      ),
      const SampleImageModel(
        id: 7,
        name: '中度雾霾-道路',
        url: 'https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800',
        category: SampleCategory.medium,
        difficulty: DifficultyLevel.medium,
        sceneType: '道路',
      ),
      const SampleImageModel(
        id: 8,
        name: '中度雾霾-森林',
        url: 'https://images.unsplash.com/photo-1448375240586-882707db888b?w=800',
        category: SampleCategory.medium,
        difficulty: DifficultyLevel.medium,
        sceneType: '森林',
      ),
      const SampleImageModel(
        id: 9,
        name: '中度雾霾-海岸',
        url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800',
        category: SampleCategory.medium,
        difficulty: DifficultyLevel.medium,
        sceneType: '海岸',
      ),
      const SampleImageModel(
        id: 10,
        name: '中度雾霾-乡村',
        url: 'https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=800',
        category: SampleCategory.medium,
        difficulty: DifficultyLevel.medium,
        sceneType: '乡村',
      ),
      // 重度雾霾
      const SampleImageModel(
        id: 11,
        name: '重度雾霾-城市中心',
        url: 'https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800',
        category: SampleCategory.heavy,
        difficulty: DifficultyLevel.hard,
        sceneType: '城市',
      ),
      const SampleImageModel(
        id: 12,
        name: '重度雾霾-高速公路',
        url: 'https://images.unsplash.com/photo-1465447142348-e9952c393450?w=800',
        category: SampleCategory.heavy,
        difficulty: DifficultyLevel.hard,
        sceneType: '道路',
      ),
      const SampleImageModel(
        id: 13,
        name: '重度雾霾-山区',
        url: 'https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800',
        category: SampleCategory.heavy,
        difficulty: DifficultyLevel.hard,
        sceneType: '山区',
      ),
      const SampleImageModel(
        id: 14,
        name: '重度雾霾-港口',
        url: 'https://images.unsplash.com/photo-1518837695005-2083093ee35b?w=800',
        category: SampleCategory.heavy,
        difficulty: DifficultyLevel.hard,
        sceneType: '港口',
      ),
      const SampleImageModel(
        id: 15,
        name: '重度雾霾-工业区',
        url: 'https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?w=800',
        category: SampleCategory.heavy,
        difficulty: DifficultyLevel.hard,
        sceneType: '工业',
      ),
      // 特殊场景
      const SampleImageModel(
        id: 16,
        name: '特殊场景-夜景雾霾',
        url: 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800',
        category: SampleCategory.special,
        difficulty: DifficultyLevel.hard,
        sceneType: '夜景',
      ),
      const SampleImageModel(
        id: 17,
        name: '特殊场景-逆光雾霾',
        url: 'https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=800',
        category: SampleCategory.special,
        difficulty: DifficultyLevel.hard,
        sceneType: '逆光',
      ),
      const SampleImageModel(
        id: 18,
        name: '特殊场景-雨雾',
        url: 'https://images.unsplash.com/photo-1428908728789-d2de25dbd4e2?w=800',
        category: SampleCategory.special,
        difficulty: DifficultyLevel.medium,
        sceneType: '雨雾',
      ),
      const SampleImageModel(
        id: 19,
        name: '特殊场景-晨雾',
        url: 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=800',
        category: SampleCategory.special,
        difficulty: DifficultyLevel.easy,
        sceneType: '晨雾',
      ),
      const SampleImageModel(
        id: 20,
        name: '特殊场景-雪雾',
        url: 'https://images.unsplash.com/photo-1491002052546-bf38f186af56?w=800',
        category: SampleCategory.special,
        difficulty: DifficultyLevel.medium,
        sceneType: '雪雾',
      ),
    ];

    if (category == null || category == SampleCategory.all) {
      return allSamples;
    }

    return allSamples.where((s) => s.category == category).toList();
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
