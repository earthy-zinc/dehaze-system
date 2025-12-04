import 'package:json_annotation/json_annotation.dart';

part 'dataset_model.g.dart';

enum ImageType {
  @JsonValue('foggy')
  foggy,
  @JsonValue('clear')
  clear,
  @JsonValue('annotated')
  annotated,
}

extension ImageTypeExtension on ImageType {
  String get displayName {
    switch (this) {
      case ImageType.foggy:
        return '有雾';
      case ImageType.clear:
        return '无雾';
      case ImageType.annotated:
        return '标注';
    }
  }
}

@JsonSerializable()
class DatasetModel {
  const DatasetModel({
    required this.id,
    required this.name,
    required this.creator,
    required this.thumbnail,
    required this.totalImages,
    required this.foggyCount,
    required this.clearCount,
    required this.annotatedCount,
    required this.createdAt,
    required this.updatedAt,
    this.description,
  });

  factory DatasetModel.fromJson(Map<String, dynamic> json) =>
      _$DatasetModelFromJson(json);
  @JsonKey(name: 'id')
  final int id;

  @JsonKey(name: 'name')
  final String name;

  @JsonKey(name: 'description')
  final String? description;

  @JsonKey(name: 'creator')
  final String creator;

  @JsonKey(name: 'thumbnail')
  final String thumbnail;

  @JsonKey(name: 'total_images')
  final int totalImages;

  @JsonKey(name: 'foggy_count')
  final int foggyCount;

  @JsonKey(name: 'clear_count')
  final int clearCount;

  @JsonKey(name: 'annotated_count')
  final int annotatedCount;

  @JsonKey(name: 'created_at')
  final DateTime createdAt;

  @JsonKey(name: 'updated_at')
  final DateTime updatedAt;

  Map<String, dynamic> toJson() => _$DatasetModelToJson(this);
}

@JsonSerializable()
class PaginatedDatasetResponse {
  const PaginatedDatasetResponse({
    required this.list,
    required this.total,
    required this.page,
    required this.pageSize,
    required this.totalPages,
  });

  factory PaginatedDatasetResponse.fromJson(Map<String, dynamic> json) =>
      _$PaginatedDatasetResponseFromJson(json);
  @JsonKey(name: 'list')
  final List<DatasetModel> list;

  @JsonKey(name: 'total')
  final int total;

  @JsonKey(name: 'page')
  final int page;

  @JsonKey(name: 'page_size')
  final int pageSize;

  @JsonKey(name: 'total_pages')
  final int totalPages;

  Map<String, dynamic> toJson() => _$PaginatedDatasetResponseToJson(this);
}

@JsonSerializable()
class ImageModel {
  const ImageModel({
    required this.id,
    required this.datasetId,
    required this.filename,
    required this.imageUrl,
    required this.imageType,
    required this.width,
    required this.height,
    required this.createdAt,
    this.fileSize,
    this.tags,
    this.description,
  });

  factory ImageModel.fromJson(Map<String, dynamic> json) =>
      _$ImageModelFromJson(json);
  @JsonKey(name: 'id')
  final int id;

  @JsonKey(name: 'dataset_id')
  final int datasetId;

  @JsonKey(name: 'filename')
  final String filename;

  @JsonKey(name: 'image_url')
  final String imageUrl;

  @JsonKey(name: 'image_type')
  final ImageType imageType;

  @JsonKey(name: 'width')
  final int width;

  @JsonKey(name: 'height')
  final int height;

  @JsonKey(name: 'file_size')
  final int? fileSize;

  @JsonKey(name: 'tags')
  final String? tags;

  @JsonKey(name: 'description')
  final String? description;

  @JsonKey(name: 'created_at')
  final DateTime createdAt;

  Map<String, dynamic> toJson() => _$ImageModelToJson(this);
}

@JsonSerializable()
class PaginatedImageResponse {
  const PaginatedImageResponse({
    required this.list,
    required this.total,
    required this.page,
    required this.pageSize,
    required this.totalPages,
  });

  factory PaginatedImageResponse.fromJson(Map<String, dynamic> json) =>
      _$PaginatedImageResponseFromJson(json);
  @JsonKey(name: 'list')
  final List<ImageModel> list;

  @JsonKey(name: 'total')
  final int total;

  @JsonKey(name: 'page')
  final int page;

  @JsonKey(name: 'page_size')
  final int pageSize;

  @JsonKey(name: 'total_pages')
  final int totalPages;

  Map<String, dynamic> toJson() => _$PaginatedImageResponseToJson(this);
}
