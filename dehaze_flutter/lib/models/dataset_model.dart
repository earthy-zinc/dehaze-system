import 'package:json_annotation/json_annotation.dart';

part 'dataset_model.g.dart';

// ==================== 数据集相关类型 ====================

@JsonSerializable()
class DatasetQuery {
  const DatasetQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keyword,
    this.type,
    this.status,
  });

  factory DatasetQuery.fromJson(Map<String, dynamic> json) =>
      _$DatasetQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? keyword;
  final String? type;
  final String? status;

  Map<String, dynamic> toJson() => _$DatasetQueryToJson(this);
}

@JsonSerializable()
class DatasetAddForm {
  const DatasetAddForm({
    required this.parentId,
    this.type,
    this.name,
    this.description,
    this.path,
    this.status,
  });

  factory DatasetAddForm.fromJson(Map<String, dynamic> json) =>
      _$DatasetAddFormFromJson(json);

  final int parentId;
  final String? type;
  final String? name;
  final String? description;
  final String? path;
  final String? status;

  Map<String, dynamic> toJson() => _$DatasetAddFormToJson(this);
}

@JsonSerializable()
class DatasetUpdateForm {
  const DatasetUpdateForm({
    this.type,
    this.name,
    this.description,
    this.path,
    this.status,
  });

  factory DatasetUpdateForm.fromJson(Map<String, dynamic> json) =>
      _$DatasetUpdateFormFromJson(json);

  final String? type;
  final String? name;
  final String? description;
  final String? path;
  final String? status;

  Map<String, dynamic> toJson() => _$DatasetUpdateFormToJson(this);
}

@JsonSerializable()
class DatasetStatistics {
  const DatasetStatistics({
    this.itemCount = 0,
    this.fileCount = 0,
    this.totalSize = 0,
    this.annotatedCount = 0,
    this.unannotatedCount = 0,
    this.sceneDistribution,
    this.hazeDistribution,
    this.formatDistribution,
    this.resolutionDistribution,
  });

  factory DatasetStatistics.fromJson(Map<String, dynamic> json) =>
      _$DatasetStatisticsFromJson(json);

  final int itemCount;
  final int fileCount;
  final int totalSize;
  final int annotatedCount;
  final int unannotatedCount;
  final Map<String, int>? sceneDistribution;
  final Map<String, int>? hazeDistribution;
  final Map<String, int>? formatDistribution;
  final Map<String, int>? resolutionDistribution;

  Map<String, dynamic> toJson() => _$DatasetStatisticsToJson(this);
}

@JsonSerializable()
class Dataset {
  const Dataset({
    required this.id,
    this.parentId,
    required this.type,
    required this.name,
    this.description,
    required this.path,
    this.hasChildren,
    this.children,
    this.status,
    this.statistics,
    this.total,
    this.createTime,
    this.updateTime,
  });

  factory Dataset.fromJson(Map<String, dynamic> json) =>
      _$DatasetFromJson(json);

  final int id;
  final int? parentId;
  final String type;
  final String name;
  final String? description;
  final String path;
  final bool? hasChildren;
  final List<Dataset>? children;
  final int? status;
  final DatasetStatistics? statistics;
  final int? total;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DatasetToJson(this);
}

// ==================== 数据项相关类型 ====================

@JsonSerializable()
class DatasetItemQuery {
  const DatasetItemQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.datasetId,
    this.keyword,
    this.sceneType,
    this.hazeLevel,
    this.minWidth,
    this.maxWidth,
    this.minHeight,
    this.maxHeight,
    this.minSize,
    this.maxSize,
    this.sortBy,
    this.sortOrder,
  });

  factory DatasetItemQuery.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final int? datasetId;
  final String? keyword;
  final String? sceneType;
  final String? hazeLevel;
  final int? minWidth;
  final int? maxWidth;
  final int? minHeight;
  final int? maxHeight;
  final int? minSize;
  final int? maxSize;
  final String? sortBy;
  final String? sortOrder;

  Map<String, dynamic> toJson() => _$DatasetItemQueryToJson(this);
}

@JsonSerializable()
class DatasetItemCreateForm {
  const DatasetItemCreateForm({
    required this.datasetId,
    this.name,
    this.sceneType,
    this.description,
  });

  factory DatasetItemCreateForm.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemCreateFormFromJson(json);

  final int datasetId;
  final String? name;
  final String? sceneType;
  final String? description;

  Map<String, dynamic> toJson() => _$DatasetItemCreateFormToJson(this);
}

@JsonSerializable()
class DatasetItemUpdateForm {
  const DatasetItemUpdateForm({
    this.name,
    this.sceneType,
  });

  factory DatasetItemUpdateForm.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemUpdateFormFromJson(json);

  final String? name;
  final String? sceneType;

  Map<String, dynamic> toJson() => _$DatasetItemUpdateFormToJson(this);
}

@JsonSerializable()
class DatasetItemSimpleVO {
  const DatasetItemSimpleVO({
    required this.id,
    required this.datasetId,
    required this.name,
    this.sceneType,
    this.description,
  });

  factory DatasetItemSimpleVO.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemSimpleVOFromJson(json);

  final int id;
  final int datasetId;
  final String name;
  final String? sceneType;
  final String? description;

  Map<String, dynamic> toJson() => _$DatasetItemSimpleVOToJson(this);
}

@JsonSerializable()
class DatasetItemVO {
  const DatasetItemVO({
    required this.id,
    required this.datasetId,
    required this.name,
    this.sceneType,
    this.description,
    this.usageCount,
    this.imageCount,
    this.clearImage,
    this.hazyImages,
    this.createTime,
    this.updateTime,
  });

  factory DatasetItemVO.fromJson(Map<String, dynamic> json) =>
      _$DatasetItemVOFromJson(json);

  final int id;
  final int datasetId;
  final String name;
  final String? sceneType;
  final String? description;
  final int? usageCount;
  final int? imageCount;
  final ImageUrlVO? clearImage;
  final List<ImageUrlVO>? hazyImages;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DatasetItemVOToJson(this);
}

// ==================== 图片文件相关类型 ====================

@JsonSerializable()
class ItemFileUpdateForm {
  const ItemFileUpdateForm({
    this.type,
    this.sceneType,
    this.hazeLevel,
    this.description,
  });

  factory ItemFileUpdateForm.fromJson(Map<String, dynamic> json) =>
      _$ItemFileUpdateFormFromJson(json);

  final String? type;
  final String? sceneType;
  final String? hazeLevel;
  final String? description;

  Map<String, dynamic> toJson() => _$ItemFileUpdateFormToJson(this);
}

@JsonSerializable()
class SimpleImageUrlVO {
  const SimpleImageUrlVO({
    required this.id,
    required this.itemId,
    required this.datasetId,
    required this.type,
    required this.url,
    this.thumbnailUrl,
    this.description,
    this.width,
    this.height,
    this.hazeLevel,
    this.fileName,
    this.sizeBytes,
    this.formattedSize,
    this.format,
    this.createTime,
  });

  factory SimpleImageUrlVO.fromJson(Map<String, dynamic> json) =>
      _$SimpleImageUrlVOFromJson(json);

  final int id;
  final int itemId;
  final int datasetId;
  final String type;
  final String url;
  final String? thumbnailUrl;
  final String? description;
  final int? width;
  final int? height;
  final String? hazeLevel;
  final String? fileName;
  final int? sizeBytes;
  final String? formattedSize;
  final String? format;
  final String? createTime;

  Map<String, dynamic> toJson() => _$SimpleImageUrlVOToJson(this);
}

@JsonSerializable()
class ImageUrlVO {
  const ImageUrlVO({
    required this.id,
    required this.itemId,
    required this.datasetId,
    this.datasetName,
    this.datasetItem,
    required this.type,
    required this.url,
    this.originUrl,
    this.thumbnailUrl,
    this.description,
    this.width,
    this.height,
    this.sceneType,
    this.hazeLevel,
    this.fileName,
    this.sizeBytes,
    this.formattedSize,
    this.format,
    this.md5,
    this.usageCount,
    this.createTime,
    this.hasPairedImages,
    this.pairedFiles,
    this.pairedCount,
  });

  factory ImageUrlVO.fromJson(Map<String, dynamic> json) =>
      _$ImageUrlVOFromJson(json);

  final int id;
  final int itemId;
  final int datasetId;
  final String? datasetName;
  final DatasetItemSimpleVO? datasetItem;
  final String type;
  final String url;
  final String? originUrl;
  final String? thumbnailUrl;
  final String? description;
  final int? width;
  final int? height;
  final String? sceneType;
  final String? hazeLevel;
  final String? fileName;
  final int? sizeBytes;
  final String? formattedSize;
  final String? format;
  final String? md5;
  final int? usageCount;
  final String? createTime;
  final bool? hasPairedImages;
  final List<SimpleImageUrlVO>? pairedFiles;
  final int? pairedCount;

  Map<String, dynamic> toJson() => _$ImageUrlVOToJson(this);
}

// ==================== 批量操作相关类型 ====================

@JsonSerializable()
class BatchDeleteForm {
  const BatchDeleteForm({
    required this.ids,
  });

  factory BatchDeleteForm.fromJson(Map<String, dynamic> json) =>
      _$BatchDeleteFormFromJson(json);

  final List<int> ids;

  Map<String, dynamic> toJson() => _$BatchDeleteFormToJson(this);
}

@JsonSerializable()
class FailedItem {
  const FailedItem({
    this.id,
    required this.reason,
  });

  factory FailedItem.fromJson(Map<String, dynamic> json) =>
      _$FailedItemFromJson(json);

  final int? id;
  final String reason;

  Map<String, dynamic> toJson() => _$FailedItemToJson(this);
}

@JsonSerializable()
class BatchDeleteResultVO {
  const BatchDeleteResultVO({
    required this.successIds,
    required this.failedItems,
    required this.successCount,
    required this.failedCount,
  });

  factory BatchDeleteResultVO.fromJson(Map<String, dynamic> json) =>
      _$BatchDeleteResultVOFromJson(json);

  final List<int> successIds;
  final List<FailedItem> failedItems;
  final int successCount;
  final int failedCount;

  Map<String, dynamic> toJson() => _$BatchDeleteResultVOToJson(this);
}

@JsonSerializable()
class BatchUploadSuccessItemVO {
  const BatchUploadSuccessItemVO({
    required this.id,
    required this.name,
    required this.fileCount,
  });

  factory BatchUploadSuccessItemVO.fromJson(Map<String, dynamic> json) =>
      _$BatchUploadSuccessItemVOFromJson(json);

  final int id;
  final String name;
  final int fileCount;

  Map<String, dynamic> toJson() => _$BatchUploadSuccessItemVOToJson(this);
}

@JsonSerializable()
class BatchUploadFailedItemVO {
  const BatchUploadFailedItemVO({
    required this.fileName,
    required this.reason,
  });

  factory BatchUploadFailedItemVO.fromJson(Map<String, dynamic> json) =>
      _$BatchUploadFailedItemVOFromJson(json);

  final String fileName;
  final String reason;

  Map<String, dynamic> toJson() => _$BatchUploadFailedItemVOToJson(this);
}

@JsonSerializable()
class BatchUploadResultVO {
  const BatchUploadResultVO({
    required this.total,
    required this.succeeded,
    required this.failed,
    required this.successItems,
    required this.failedItems,
  });

  factory BatchUploadResultVO.fromJson(Map<String, dynamic> json) =>
      _$BatchUploadResultVOFromJson(json);

  final int total;
  final int succeeded;
  final int failed;
  final List<BatchUploadSuccessItemVO> successItems;
  final List<BatchUploadFailedItemVO> failedItems;

  Map<String, dynamic> toJson() => _$BatchUploadResultVOToJson(this);
}

@JsonSerializable()
class BatchActionFailureDetailVO {
  const BatchActionFailureDetailVO({
    this.identifier,
    required this.reason,
  });

  factory BatchActionFailureDetailVO.fromJson(Map<String, dynamic> json) =>
      _$BatchActionFailureDetailVOFromJson(json);

  final String? identifier;
  final String reason;

  Map<String, dynamic> toJson() => _$BatchActionFailureDetailVOToJson(this);
}

@JsonSerializable()
class BatchOperationResultVO {
  const BatchOperationResultVO({
    required this.successCount,
    required this.failedCount,
    required this.message,
    this.successIds,
    this.failureDetails,
  });

  factory BatchOperationResultVO.fromJson(Map<String, dynamic> json) =>
      _$BatchOperationResultVOFromJson(json);

  final int successCount;
  final int failedCount;
  final String message;
  final List<int>? successIds;
  final List<BatchActionFailureDetailVO>? failureDetails;

  Map<String, dynamic> toJson() => _$BatchOperationResultVOToJson(this);
}
