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
}

/// 数据集模型
@JsonSerializable()
class DatasetModel {
  const DatasetModel({
    required this.id,
    required this.name,
    required this.createTime,
    this.parentId,
    this.type,
    this.path,
    this.description,
    this.remark,
    this.usageCount,
    this.createBy,
    this.updateTime,
    this.updateBy,
    this.children = const [],
    this.status,
  });

  factory DatasetModel.fromJson(Map<String, dynamic> json) =>
      _$DatasetModelFromJson(json);

  final int id;

  @JsonKey(name: 'parentId')
  final int? parentId;

  final String name;

  /// 数据集类型（如 indoor、outdoor）
  final String? type;

  /// 存储路径
  final String? path;

  final String? description;
  final String? remark;

  /// 使用次数
  @JsonKey(name: 'usageCount')
  final int? usageCount;

  @JsonKey(name: 'createBy')
  final String? createBy;

  @JsonKey(name: 'createTime')
  final String createTime;

  @JsonKey(name: 'updateBy')
  final String? updateBy;

  @JsonKey(name: 'updateTime')
  final String? updateTime;

  /// 子数据集（树形结构）
  final List<DatasetModel> children;

  /// 状态（1=启用 0=禁用）
  final int? status;

  Map<String, dynamic> toJson() => _$DatasetModelToJson(this);

  /// 是否有子数据集
  bool get hasChildren => children.isNotEmpty;
}

/// 数据项模型
@JsonSerializable()
class DatasetItemModel {
  const DatasetItemModel({
    required this.id,
    required this.datasetId,
    required this.createTime,
    this.name,
    this.description,
    this.files = const [],
  });

  factory DatasetItemModel.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemModelFromJson(json);

  final int id;

  @JsonKey(name: 'datasetId')
  final int datasetId;

  final String? name;
  final String? description;

  /// 关联的图片文件列表
  final List<ItemFileModel> files;

  @JsonKey(name: 'createTime')
  final String createTime;

  Map<String, dynamic> toJson() => _$DatasetItemModelToJson(this);
}

/// 数据项图片文件
@JsonSerializable()
class ItemFileModel {
  const ItemFileModel({
    required this.id,
    required this.itemId,
    required this.fileType,
    required this.fileUrl,
    this.fileId,
    this.fileName,
    this.fileSize,
    this.width,
    this.height,
  });

  factory ItemFileModel.fromJson(Map<String, dynamic> json) =>
      _$ItemFileModelFromJson(json);

  final int id;

  @JsonKey(name: 'itemId')
  final int itemId;

  @JsonKey(name: 'fileId')
  final String? fileId;

  /// 图片类型（hazy/clear/dehazed）
  @JsonKey(name: 'fileType')
  final String fileType;

  @JsonKey(name: 'fileUrl')
  final String fileUrl;

  @JsonKey(name: 'fileName')
  final String? fileName;

  @JsonKey(name: 'fileSize')
  final int? fileSize;

  final int? width;
  final int? height;

  Map<String, dynamic> toJson() => _$ItemFileModelToJson(this);

  /// 转换为 ImageType 枚举
  ImageType get imageType {
    switch (fileType) {
      case 'hazy':
        return ImageType.hazy;
      case 'clear':
        return ImageType.clear;
      case 'dehazed':
        return ImageType.dehazed;
      default:
        return ImageType.hazy;
    }
  }
}

/// 图片展示模型（前端使用）
@JsonSerializable()
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
    this.tags,
    this.description,
  });

  factory ImageModel.fromJson(Map<String, dynamic> json) =>
      _$ImageModelFromJson(json);

  final int id;

  @JsonKey(name: 'datasetId')
  final int datasetId;

  final String filename;

  @JsonKey(name: 'fileUrl')
  final String imageUrl;

  @JsonKey(name: 'fileType')
  @JsonKey(unknownEnumValue: ImageType.hazy)
  final ImageType imageType;

  final int? width;
  final int? height;
  final int? fileSize;
  final String? tags;
  final String? description;

  @JsonKey(name: 'createTime')
  final String createdAt;

  Map<String, dynamic> toJson() => _$ImageModelToJson(this);
}
