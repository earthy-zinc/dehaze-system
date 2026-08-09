import 'package:json_annotation/json_annotation.dart';

part 'dataset_model.g.dart';

/// 图片类型枚举（与后端一致）
enum ImageType {
  @JsonValue('hazy')
  hazy,
  @JsonValue('clear')
  clear,
  @JsonValue('dehazed')
  dehazed,
}

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

/// 数据项模型
///
/// 对应后端 DatasetItemVO：清晰图（clearImage）+ 有雾图列表（hazyImages）。
@JsonSerializable()
class DatasetItemModel {
  const DatasetItemModel({
    required this.id,
    required this.datasetId,
    required this.createTime,
    this.name,
    this.sceneType,
    this.description,
    this.usageCount,
    this.imageCount,
    this.clearImage,
    this.hazyImages = const [],
    this.updateTime,
  });

  factory DatasetItemModel.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemModelFromJson(json);

  final int id;

  @JsonKey(name: 'datasetId')
  final int datasetId;

  final String? name;

  /// 场景类型（如：城市街道、山区风景）
  @JsonKey(name: 'sceneType')
  final String? sceneType;

  final String? description;

  /// 使用次数
  @JsonKey(name: 'usageCount')
  final int? usageCount;

  /// 图片总数（清晰图 + 有雾图）
  @JsonKey(name: 'imageCount')
  final int? imageCount;

  /// 清晰图（Ground Truth），每个数据项只有一张
  @JsonKey(name: 'clearImage')
  final ItemImageModel? clearImage;

  /// 有雾图列表，每个数据项可以有多张不同雾霾程度的有雾图
  @JsonKey(name: 'hazyImages')
  final List<ItemImageModel> hazyImages;

  @JsonKey(name: 'createTime')
  final String createTime;

  @JsonKey(name: 'updateTime')
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DatasetItemModelToJson(this);

  /// 数据项下的全部图片（清晰图 + 有雾图）
  List<ItemImageModel> get allImages => [
        if (clearImage != null) clearImage!,
        ...hazyImages,
      ];
}

/// 数据项图片
///
/// 对应后端 ImageUrlVO。
@JsonSerializable()
class ItemImageModel {
  const ItemImageModel({
    required this.id,
    required this.url,
    this.itemId,
    this.datasetId,
    this.type,
    this.originUrl,
    this.thumbnailUrl,
    this.fileName,
    this.width,
    this.height,
    this.sizeBytes,
    this.hazeLevel,
    this.sceneType,
    this.description,
    this.createTime,
  });

  factory ItemImageModel.fromJson(Map<String, dynamic> json) =>
      _$ItemImageModelFromJson(json);

  /// 数据项文件 ID
  final int id;

  @JsonKey(name: 'itemId')
  final int? itemId;

  @JsonKey(name: 'datasetId')
  final int? datasetId;

  /// 图片类型：clear-清晰图，hazy-有雾图
  final String? type;

  /// 图片访问 URL
  final String url;

  /// 原始图片 URL
  @JsonKey(name: 'originUrl')
  final String? originUrl;

  /// 缩略图 URL
  @JsonKey(name: 'thumbnailUrl')
  final String? thumbnailUrl;

  @JsonKey(name: 'fileName')
  final String? fileName;

  final int? width;
  final int? height;

  /// 文件大小（字节）
  @JsonKey(name: 'sizeBytes')
  final int? sizeBytes;

  /// 雾霾程度：light/medium/heavy
  @JsonKey(name: 'hazeLevel')
  final String? hazeLevel;

  @JsonKey(name: 'sceneType')
  final String? sceneType;

  final String? description;

  @JsonKey(name: 'createTime')
  final String? createTime;

  Map<String, dynamic> toJson() => _$ItemImageModelToJson(this);

  /// 转换为 ImageType 枚举
  ImageType get imageType => ImageTypeExtension.fromValue(type);
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

  /// 从数据项图片构建展示模型
  factory ImageModel.fromItemImage(ItemImageModel image, int datasetId, String createdAt) =>
      ImageModel(
        id: image.id,
        datasetId: datasetId,
        filename: image.fileName ?? 'image_${image.id}',
        imageUrl: image.url,
        imageType: image.imageType,
        createdAt: createdAt,
        width: image.width,
        height: image.height,
        fileSize: image.sizeBytes,
      );

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
