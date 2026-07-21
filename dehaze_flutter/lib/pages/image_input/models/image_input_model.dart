import 'dart:typed_data';

import 'package:json_annotation/json_annotation.dart';

part 'image_input_model.g.dart';

/// 输入方式枚举
enum InputMethod {
  @JsonValue('upload')
  upload, // 上传图片
  @JsonValue('camera')
  camera, // 拍照
  @JsonValue('sample')
  sample, // 样例图片
  @JsonValue('history')
  history, // 历史记录
}

extension InputMethodExtension on InputMethod {
  String get displayName {
    switch (this) {
      case InputMethod.upload:
        return '上传图片';
      case InputMethod.camera:
        return '拍照';
      case InputMethod.sample:
        return '样例图片';
      case InputMethod.history:
        return '历史记录';
    }
  }

  String get subtitle {
    switch (this) {
      case InputMethod.upload:
        return '从相册选择';
      case InputMethod.camera:
        return '实时拍摄';
      case InputMethod.sample:
        return '快速体验';
      case InputMethod.history:
        return '最近处理';
    }
  }
}

/// 样例图片分类
enum SampleCategory {
  @JsonValue('all')
  all, // 全部
  @JsonValue('light')
  light, // 轻度雾霾
  @JsonValue('medium')
  medium, // 中度雾霾
  @JsonValue('heavy')
  heavy, // 重度雾霾
  @JsonValue('special')
  special, // 特殊场景
}

extension SampleCategoryExtension on SampleCategory {
  String get displayName {
    switch (this) {
      case SampleCategory.all:
        return '全部';
      case SampleCategory.light:
        return '轻度雾霾';
      case SampleCategory.medium:
        return '中度雾霾';
      case SampleCategory.heavy:
        return '重度雾霾';
      case SampleCategory.special:
        return '特殊场景';
    }
  }
}

/// 难度等级
enum DifficultyLevel {
  @JsonValue('easy')
  easy, // 简单
  @JsonValue('medium')
  medium, // 中等
  @JsonValue('hard')
  hard, // 困难
}

extension DifficultyLevelExtension on DifficultyLevel {
  String get displayName {
    switch (this) {
      case DifficultyLevel.easy:
        return '简单';
      case DifficultyLevel.medium:
        return '中等';
      case DifficultyLevel.hard:
        return '困难';
    }
  }
}

/// 图片来源
enum ImageSource {
  @JsonValue('upload')
  upload, // 上传
  @JsonValue('camera')
  camera, // 拍照
  @JsonValue('sample')
  sample, // 样例
  @JsonValue('history')
  history, // 历史
}

/// 选中的图片模型
@JsonSerializable()
class SelectedImageModel {
  const SelectedImageModel({
    required this.id,
    required this.url,
    required this.filename,
    required this.width,
    required this.height,
    required this.fileSize,
    required this.source,
    this.localPath,
    this.bytes,
    this.sampleInfo,
  });

  factory SelectedImageModel.fromJson(Map<String, dynamic> json) =>
      _$SelectedImageModelFromJson(json);

  @JsonKey(name: 'id')
  final String id;

  @JsonKey(name: 'url')
  final String url; // 图片URL或本地路径

  @JsonKey(name: 'local_path')
  final String? localPath; // 本地文件路径

  /// 图片字节流（内存态，跨平台渲染本地图片，不参与序列化）
  @JsonKey(includeFromJson: false, includeToJson: false)
  final Uint8List? bytes;

  @JsonKey(name: 'filename')
  final String filename;

  @JsonKey(name: 'width')
  final int width;

  @JsonKey(name: 'height')
  final int height;

  @JsonKey(name: 'file_size')
  final int fileSize; // 字节

  @JsonKey(name: 'source')
  final ImageSource source; // 图片来源

  @JsonKey(name: 'sample_info')
  final SampleImageModel? sampleInfo; // 样例图片信息（可选）

  Map<String, dynamic> toJson() => _$SelectedImageModelToJson(this);

  SelectedImageModel copyWith({
    String? id,
    String? url,
    String? localPath,
    Uint8List? bytes,
    String? filename,
    int? width,
    int? height,
    int? fileSize,
    ImageSource? source,
    SampleImageModel? sampleInfo,
  }) =>
      SelectedImageModel(
        id: id ?? this.id,
        url: url ?? this.url,
        localPath: localPath ?? this.localPath,
        bytes: bytes ?? this.bytes,
        filename: filename ?? this.filename,
        width: width ?? this.width,
        height: height ?? this.height,
        fileSize: fileSize ?? this.fileSize,
        source: source ?? this.source,
        sampleInfo: sampleInfo ?? this.sampleInfo,
      );
}

/// 样例图片模型
@JsonSerializable()
class SampleImageModel {
  const SampleImageModel({
    required this.id,
    required this.name,
    required this.url,
    required this.category,
    required this.difficulty,
    this.sceneType,
    this.recommendedAlgorithm,
  });

  factory SampleImageModel.fromJson(Map<String, dynamic> json) =>
      _$SampleImageModelFromJson(json);

  @JsonKey(name: 'id')
  final int id;

  @JsonKey(name: 'name')
  final String name;

  @JsonKey(name: 'url')
  final String url;

  @JsonKey(name: 'category')
  final SampleCategory category;

  @JsonKey(name: 'difficulty')
  final DifficultyLevel difficulty;

  @JsonKey(name: 'scene_type')
  final String? sceneType; // 场景类型

  @JsonKey(name: 'recommended_algorithm')
  final String? recommendedAlgorithm;

  Map<String, dynamic> toJson() => _$SampleImageModelToJson(this);
}

/// 样例图片分页响应
@JsonSerializable()
class PaginatedSampleResponse {
  const PaginatedSampleResponse({
    required this.list,
    required this.total,
    required this.page,
    required this.pageSize,
    required this.totalPages,
  });

  factory PaginatedSampleResponse.fromJson(Map<String, dynamic> json) =>
      _$PaginatedSampleResponseFromJson(json);

  @JsonKey(name: 'list')
  final List<SampleImageModel> list;

  @JsonKey(name: 'total')
  final int total;

  @JsonKey(name: 'page')
  final int page;

  @JsonKey(name: 'page_size')
  final int pageSize;

  @JsonKey(name: 'total_pages')
  final int totalPages;

  Map<String, dynamic> toJson() => _$PaginatedSampleResponseToJson(this);
}

/// 历史记录模型
@JsonSerializable()
class HistoryRecordModel {
  const HistoryRecordModel({
    required this.id,
    required this.originalThumbnail,
    required this.filename,
    required this.timestamp,
    required this.isSuccess,
    this.resultThumbnail,
    this.algorithmName,
    this.parameters,
  });

  factory HistoryRecordModel.fromJson(Map<String, dynamic> json) =>
      _$HistoryRecordModelFromJson(json);

  @JsonKey(name: 'id')
  final String id;

  @JsonKey(name: 'original_thumbnail')
  final String originalThumbnail;

  @JsonKey(name: 'result_thumbnail')
  final String? resultThumbnail;

  @JsonKey(name: 'filename')
  final String filename;

  @JsonKey(name: 'timestamp')
  final DateTime timestamp;

  @JsonKey(name: 'algorithm_name')
  final String? algorithmName;

  @JsonKey(name: 'parameters')
  final Map<String, dynamic>? parameters;

  @JsonKey(name: 'is_success')
  final bool isSuccess;

  Map<String, dynamic> toJson() => _$HistoryRecordModelToJson(this);
}

/// 图片验证结果
class ImageValidationResult {
  const ImageValidationResult({
    required this.isValid,
    this.errorMessage,
    this.needsCompression = false,
  });

  final bool isValid;
  final String? errorMessage;
  final bool needsCompression;
}

/// 上传进度状态
class UploadProgress {
  const UploadProgress({
    required this.progress,
    required this.status,
    this.errorMessage,
  });

  final double progress; // 0.0 - 1.0
  final UploadStatus status;
  final String? errorMessage;

  static const idle = UploadProgress(progress: 0, status: UploadStatus.idle);
}

enum UploadStatus {
  idle,
  selecting,
  validating,
  compressing,
  uploading,
  success,
  error,
}
