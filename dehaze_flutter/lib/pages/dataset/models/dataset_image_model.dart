/// 数据集页面私有展示模型
///
/// 后端 DTO（DatasetItemVO / ImageUrlVO）在 lib/models/dataset_model.dart；
/// 本文件只放页面展示所需的模型，不参与接口序列化。
library;

/// 图片类型枚举（与后端一致）
enum ImageType { hazy, clear, dehazed }

extension ImageTypeExtension on ImageType {
  String get displayName {
    switch (this) {
      case ImageType.hazy:
        return '有雾';
      case ImageType.clear:
        return '清晰';
      case ImageType.dehazed:
        return '去雾结果';
    }
  }

  /// 从后端 type 字符串解析（未知类型按有雾图处理）
  static ImageType fromValue(String? value) {
    switch (value) {
      case 'clear':
        return ImageType.clear;
      case 'dehazed':
        return ImageType.dehazed;
      default:
        return ImageType.hazy;
    }
  }
}

/// 图片展示模型（前端列表/网格使用）
class ImageModel {
  const ImageModel({
    required this.id,
    required this.datasetId,
    required this.filename,
    required this.imageUrl,
    required this.imageType,
    required this.createdAt,
    this.width,
    this.height,
    this.fileSize,
  });

  final int id;
  final int datasetId;
  final String filename;
  final String imageUrl;
  final ImageType imageType;
  final String createdAt;
  final int? width;
  final int? height;
  final int? fileSize;
}
