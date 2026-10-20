// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dataset_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DatasetQuery _$DatasetQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetQuery', json, ($checkedConvert) {
      final val = DatasetQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        keyword: $checkedConvert('keyword', (v) => v as String?),
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetQueryToJson(DatasetQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'keyword': ?instance.keyword,
      'type': ?instance.type,
      'status': ?instance.status,
    };

DatasetAddForm _$DatasetAddFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetAddForm', json, ($checkedConvert) {
      final val = DatasetAddForm(
        parentId: $checkedConvert('parentId', (v) => (v as num).toInt()),
        type: $checkedConvert('type', (v) => v as String?),
        name: $checkedConvert('name', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetAddFormToJson(DatasetAddForm instance) =>
    <String, dynamic>{
      'parentId': instance.parentId,
      'type': ?instance.type,
      'name': ?instance.name,
      'description': ?instance.description,
      'path': ?instance.path,
      'status': ?instance.status,
    };

DatasetUpdateForm _$DatasetUpdateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetUpdateForm', json, ($checkedConvert) {
      final val = DatasetUpdateForm(
        type: $checkedConvert('type', (v) => v as String?),
        name: $checkedConvert('name', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetUpdateFormToJson(DatasetUpdateForm instance) =>
    <String, dynamic>{
      'type': ?instance.type,
      'name': ?instance.name,
      'description': ?instance.description,
      'path': ?instance.path,
      'status': ?instance.status,
    };

DatasetStatistics _$DatasetStatisticsFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('DatasetStatistics', json, ($checkedConvert) {
  final val = DatasetStatistics(
    itemCount: $checkedConvert('itemCount', (v) => (v as num?)?.toInt() ?? 0),
    fileCount: $checkedConvert('fileCount', (v) => (v as num?)?.toInt() ?? 0),
    totalSize: $checkedConvert('totalSize', (v) => (v as num?)?.toInt() ?? 0),
    annotatedCount: $checkedConvert(
      'annotatedCount',
      (v) => (v as num?)?.toInt() ?? 0,
    ),
    unannotatedCount: $checkedConvert(
      'unannotatedCount',
      (v) => (v as num?)?.toInt() ?? 0,
    ),
    sceneDistribution: $checkedConvert(
      'sceneDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    hazeDistribution: $checkedConvert(
      'hazeDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    formatDistribution: $checkedConvert(
      'formatDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    resolutionDistribution: $checkedConvert(
      'resolutionDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
  );
  return val;
});

Map<String, dynamic> _$DatasetStatisticsToJson(DatasetStatistics instance) =>
    <String, dynamic>{
      'itemCount': instance.itemCount,
      'fileCount': instance.fileCount,
      'totalSize': instance.totalSize,
      'annotatedCount': instance.annotatedCount,
      'unannotatedCount': instance.unannotatedCount,
      'sceneDistribution': ?instance.sceneDistribution,
      'hazeDistribution': ?instance.hazeDistribution,
      'formatDistribution': ?instance.formatDistribution,
      'resolutionDistribution': ?instance.resolutionDistribution,
    };

Dataset _$DatasetFromJson(Map<String, dynamic> json) =>
    $checkedCreate('Dataset', json, ($checkedConvert) {
      final val = Dataset(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        parentId: $checkedConvert('parentId', (v) => (v as num?)?.toInt()),
        type: $checkedConvert('type', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String),
        hasChildren: $checkedConvert('hasChildren', (v) => v as bool?),
        children: $checkedConvert(
          'children',
          (v) => (v as List<dynamic>?)
              ?.map((e) => Dataset.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        statistics: $checkedConvert(
          'statistics',
          (v) => v == null
              ? null
              : DatasetStatistics.fromJson(v as Map<String, dynamic>),
        ),
        total: $checkedConvert('total', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetToJson(Dataset instance) => <String, dynamic>{
  'id': instance.id,
  'parentId': ?instance.parentId,
  'type': instance.type,
  'name': instance.name,
  'description': ?instance.description,
  'path': instance.path,
  'hasChildren': ?instance.hasChildren,
  'children': ?instance.children?.map((e) => e.toJson()).toList(),
  'status': ?instance.status,
  'statistics': ?instance.statistics?.toJson(),
  'total': ?instance.total,
  'createTime': ?instance.createTime,
  'updateTime': ?instance.updateTime,
};

DatasetItemQuery _$DatasetItemQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetItemQuery', json, ($checkedConvert) {
      final val = DatasetItemQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        datasetId: $checkedConvert('datasetId', (v) => (v as num?)?.toInt()),
        keyword: $checkedConvert('keyword', (v) => v as String?),
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        hazeLevel: $checkedConvert('hazeLevel', (v) => v as String?),
        minWidth: $checkedConvert('minWidth', (v) => (v as num?)?.toInt()),
        maxWidth: $checkedConvert('maxWidth', (v) => (v as num?)?.toInt()),
        minHeight: $checkedConvert('minHeight', (v) => (v as num?)?.toInt()),
        maxHeight: $checkedConvert('maxHeight', (v) => (v as num?)?.toInt()),
        minSize: $checkedConvert('minSize', (v) => (v as num?)?.toInt()),
        maxSize: $checkedConvert('maxSize', (v) => (v as num?)?.toInt()),
        sortBy: $checkedConvert('sortBy', (v) => v as String?),
        sortOrder: $checkedConvert('sortOrder', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetItemQueryToJson(DatasetItemQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'datasetId': ?instance.datasetId,
      'keyword': ?instance.keyword,
      'sceneType': ?instance.sceneType,
      'hazeLevel': ?instance.hazeLevel,
      'minWidth': ?instance.minWidth,
      'maxWidth': ?instance.maxWidth,
      'minHeight': ?instance.minHeight,
      'maxHeight': ?instance.maxHeight,
      'minSize': ?instance.minSize,
      'maxSize': ?instance.maxSize,
      'sortBy': ?instance.sortBy,
      'sortOrder': ?instance.sortOrder,
    };

DatasetItemCreateForm _$DatasetItemCreateFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('DatasetItemCreateForm', json, ($checkedConvert) {
  final val = DatasetItemCreateForm(
    datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String?),
    sceneType: $checkedConvert('sceneType', (v) => v as String?),
    description: $checkedConvert('description', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$DatasetItemCreateFormToJson(
  DatasetItemCreateForm instance,
) => <String, dynamic>{
  'datasetId': instance.datasetId,
  'name': ?instance.name,
  'sceneType': ?instance.sceneType,
  'description': ?instance.description,
};

DatasetItemUpdateForm _$DatasetItemUpdateFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('DatasetItemUpdateForm', json, ($checkedConvert) {
  final val = DatasetItemUpdateForm(
    name: $checkedConvert('name', (v) => v as String?),
    sceneType: $checkedConvert('sceneType', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$DatasetItemUpdateFormToJson(
  DatasetItemUpdateForm instance,
) => <String, dynamic>{
  'name': ?instance.name,
  'sceneType': ?instance.sceneType,
};

DatasetItemSimpleVO _$DatasetItemSimpleVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetItemSimpleVO', json, ($checkedConvert) {
      final val = DatasetItemSimpleVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetItemSimpleVOToJson(
  DatasetItemSimpleVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'datasetId': instance.datasetId,
  'name': instance.name,
  'sceneType': ?instance.sceneType,
  'description': ?instance.description,
};

DatasetItemVO _$DatasetItemVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetItemVO', json, ($checkedConvert) {
      final val = DatasetItemVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        usageCount: $checkedConvert('usageCount', (v) => (v as num?)?.toInt()),
        imageCount: $checkedConvert('imageCount', (v) => (v as num?)?.toInt()),
        clearImage: $checkedConvert(
          'clearImage',
          (v) =>
              v == null ? null : ImageUrlVO.fromJson(v as Map<String, dynamic>),
        ),
        hazyImages: $checkedConvert(
          'hazyImages',
          (v) => (v as List<dynamic>?)
              ?.map((e) => ImageUrlVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$DatasetItemVOToJson(DatasetItemVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'datasetId': instance.datasetId,
      'name': instance.name,
      'sceneType': ?instance.sceneType,
      'description': ?instance.description,
      'usageCount': ?instance.usageCount,
      'imageCount': ?instance.imageCount,
      'clearImage': ?instance.clearImage?.toJson(),
      'hazyImages': ?instance.hazyImages?.map((e) => e.toJson()).toList(),
      'createTime': ?instance.createTime,
      'updateTime': ?instance.updateTime,
    };

ItemFileUpdateForm _$ItemFileUpdateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ItemFileUpdateForm', json, ($checkedConvert) {
      final val = ItemFileUpdateForm(
        type: $checkedConvert('type', (v) => v as String?),
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        hazeLevel: $checkedConvert('hazeLevel', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ItemFileUpdateFormToJson(ItemFileUpdateForm instance) =>
    <String, dynamic>{
      'type': ?instance.type,
      'sceneType': ?instance.sceneType,
      'hazeLevel': ?instance.hazeLevel,
      'description': ?instance.description,
    };

SimpleImageUrlVO _$SimpleImageUrlVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('SimpleImageUrlVO', json, ($checkedConvert) {
      final val = SimpleImageUrlVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        itemId: $checkedConvert('itemId', (v) => (v as num).toInt()),
        datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
        type: $checkedConvert('type', (v) => v as String),
        url: $checkedConvert('url', (v) => v as String),
        thumbnailUrl: $checkedConvert('thumbnailUrl', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        width: $checkedConvert('width', (v) => (v as num?)?.toInt()),
        height: $checkedConvert('height', (v) => (v as num?)?.toInt()),
        hazeLevel: $checkedConvert('hazeLevel', (v) => v as String?),
        fileName: $checkedConvert('fileName', (v) => v as String?),
        sizeBytes: $checkedConvert('sizeBytes', (v) => (v as num?)?.toInt()),
        formattedSize: $checkedConvert('formattedSize', (v) => v as String?),
        format: $checkedConvert('format', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$SimpleImageUrlVOToJson(SimpleImageUrlVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'itemId': instance.itemId,
      'datasetId': instance.datasetId,
      'type': instance.type,
      'url': instance.url,
      'thumbnailUrl': ?instance.thumbnailUrl,
      'description': ?instance.description,
      'width': ?instance.width,
      'height': ?instance.height,
      'hazeLevel': ?instance.hazeLevel,
      'fileName': ?instance.fileName,
      'sizeBytes': ?instance.sizeBytes,
      'formattedSize': ?instance.formattedSize,
      'format': ?instance.format,
      'createTime': ?instance.createTime,
    };

ImageUrlVO _$ImageUrlVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'ImageUrlVO',
  json,
  ($checkedConvert) {
    final val = ImageUrlVO(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      itemId: $checkedConvert('itemId', (v) => (v as num).toInt()),
      datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
      datasetName: $checkedConvert('datasetName', (v) => v as String?),
      datasetItem: $checkedConvert(
        'datasetItem',
        (v) => v == null
            ? null
            : DatasetItemSimpleVO.fromJson(v as Map<String, dynamic>),
      ),
      type: $checkedConvert('type', (v) => v as String),
      url: $checkedConvert('url', (v) => v as String),
      thumbnailUrl: $checkedConvert('thumbnailUrl', (v) => v as String?),
      description: $checkedConvert('description', (v) => v as String?),
      width: $checkedConvert('width', (v) => (v as num?)?.toInt()),
      height: $checkedConvert('height', (v) => (v as num?)?.toInt()),
      sceneType: $checkedConvert('sceneType', (v) => v as String?),
      hazeLevel: $checkedConvert('hazeLevel', (v) => v as String?),
      fileName: $checkedConvert('fileName', (v) => v as String?),
      sizeBytes: $checkedConvert('sizeBytes', (v) => (v as num?)?.toInt()),
      formattedSize: $checkedConvert('formattedSize', (v) => v as String?),
      format: $checkedConvert('format', (v) => v as String?),
      md5: $checkedConvert('md5', (v) => v as String?),
      usageCount: $checkedConvert('usageCount', (v) => (v as num?)?.toInt()),
      createTime: $checkedConvert('createTime', (v) => v as String?),
      hasPairedImages: $checkedConvert('hasPairedImages', (v) => v as bool?),
      pairedFiles: $checkedConvert(
        'pairedFiles',
        (v) => (v as List<dynamic>?)
            ?.map((e) => SimpleImageUrlVO.fromJson(e as Map<String, dynamic>))
            .toList(),
      ),
      pairedCount: $checkedConvert('pairedCount', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$ImageUrlVOToJson(ImageUrlVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'itemId': instance.itemId,
      'datasetId': instance.datasetId,
      'datasetName': ?instance.datasetName,
      'datasetItem': ?instance.datasetItem?.toJson(),
      'type': instance.type,
      'url': instance.url,
      'thumbnailUrl': ?instance.thumbnailUrl,
      'description': ?instance.description,
      'width': ?instance.width,
      'height': ?instance.height,
      'sceneType': ?instance.sceneType,
      'hazeLevel': ?instance.hazeLevel,
      'fileName': ?instance.fileName,
      'sizeBytes': ?instance.sizeBytes,
      'formattedSize': ?instance.formattedSize,
      'format': ?instance.format,
      'md5': ?instance.md5,
      'usageCount': ?instance.usageCount,
      'createTime': ?instance.createTime,
      'hasPairedImages': ?instance.hasPairedImages,
      'pairedFiles': ?instance.pairedFiles?.map((e) => e.toJson()).toList(),
      'pairedCount': ?instance.pairedCount,
    };

BatchDeleteForm _$BatchDeleteFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('BatchDeleteForm', json, ($checkedConvert) {
      final val = BatchDeleteForm(
        ids: $checkedConvert(
          'ids',
          (v) => (v as List<dynamic>).map((e) => (e as num).toInt()).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$BatchDeleteFormToJson(BatchDeleteForm instance) =>
    <String, dynamic>{'ids': instance.ids};

FailedItem _$FailedItemFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FailedItem', json, ($checkedConvert) {
      final val = FailedItem(
        id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
        reason: $checkedConvert('reason', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FailedItemToJson(FailedItem instance) =>
    <String, dynamic>{'id': ?instance.id, 'reason': instance.reason};

BatchDeleteResultVO _$BatchDeleteResultVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('BatchDeleteResultVO', json, ($checkedConvert) {
      final val = BatchDeleteResultVO(
        successIds: $checkedConvert(
          'successIds',
          (v) => (v as List<dynamic>).map((e) => (e as num).toInt()).toList(),
        ),
        failedItems: $checkedConvert(
          'failedItems',
          (v) => (v as List<dynamic>)
              .map((e) => FailedItem.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        successCount: $checkedConvert(
          'successCount',
          (v) => (v as num).toInt(),
        ),
        failedCount: $checkedConvert('failedCount', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$BatchDeleteResultVOToJson(
  BatchDeleteResultVO instance,
) => <String, dynamic>{
  'successIds': instance.successIds,
  'failedItems': instance.failedItems.map((e) => e.toJson()).toList(),
  'successCount': instance.successCount,
  'failedCount': instance.failedCount,
};

BatchUploadSuccessItemVO _$BatchUploadSuccessItemVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('BatchUploadSuccessItemVO', json, ($checkedConvert) {
  final val = BatchUploadSuccessItemVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    fileCount: $checkedConvert('fileCount', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$BatchUploadSuccessItemVOToJson(
  BatchUploadSuccessItemVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'fileCount': instance.fileCount,
};

BatchUploadFailedItemVO _$BatchUploadFailedItemVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('BatchUploadFailedItemVO', json, ($checkedConvert) {
  final val = BatchUploadFailedItemVO(
    fileName: $checkedConvert('fileName', (v) => v as String),
    reason: $checkedConvert('reason', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$BatchUploadFailedItemVOToJson(
  BatchUploadFailedItemVO instance,
) => <String, dynamic>{
  'fileName': instance.fileName,
  'reason': instance.reason,
};

BatchUploadResultVO _$BatchUploadResultVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('BatchUploadResultVO', json, ($checkedConvert) {
      final val = BatchUploadResultVO(
        total: $checkedConvert('total', (v) => (v as num).toInt()),
        succeeded: $checkedConvert('succeeded', (v) => (v as num).toInt()),
        failed: $checkedConvert('failed', (v) => (v as num).toInt()),
        successItems: $checkedConvert(
          'successItems',
          (v) => (v as List<dynamic>)
              .map(
                (e) => BatchUploadSuccessItemVO.fromJson(
                  e as Map<String, dynamic>,
                ),
              )
              .toList(),
        ),
        failedItems: $checkedConvert(
          'failedItems',
          (v) => (v as List<dynamic>)
              .map(
                (e) =>
                    BatchUploadFailedItemVO.fromJson(e as Map<String, dynamic>),
              )
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$BatchUploadResultVOToJson(
  BatchUploadResultVO instance,
) => <String, dynamic>{
  'total': instance.total,
  'succeeded': instance.succeeded,
  'failed': instance.failed,
  'successItems': instance.successItems.map((e) => e.toJson()).toList(),
  'failedItems': instance.failedItems.map((e) => e.toJson()).toList(),
};

BatchActionFailureDetailVO _$BatchActionFailureDetailVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('BatchActionFailureDetailVO', json, ($checkedConvert) {
  final val = BatchActionFailureDetailVO(
    identifier: $checkedConvert('identifier', (v) => v as String?),
    reason: $checkedConvert('reason', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$BatchActionFailureDetailVOToJson(
  BatchActionFailureDetailVO instance,
) => <String, dynamic>{
  'identifier': ?instance.identifier,
  'reason': instance.reason,
};

BatchOperationResultVO _$BatchOperationResultVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('BatchOperationResultVO', json, ($checkedConvert) {
  final val = BatchOperationResultVO(
    successCount: $checkedConvert('successCount', (v) => (v as num).toInt()),
    failedCount: $checkedConvert('failedCount', (v) => (v as num).toInt()),
    message: $checkedConvert('message', (v) => v as String),
    successIds: $checkedConvert(
      'successIds',
      (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
    ),
    failureDetails: $checkedConvert(
      'failureDetails',
      (v) => (v as List<dynamic>?)
          ?.map(
            (e) =>
                BatchActionFailureDetailVO.fromJson(e as Map<String, dynamic>),
          )
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$BatchOperationResultVOToJson(
  BatchOperationResultVO instance,
) => <String, dynamic>{
  'successCount': instance.successCount,
  'failedCount': instance.failedCount,
  'message': instance.message,
  'successIds': ?instance.successIds,
  'failureDetails': ?instance.failureDetails?.map((e) => e.toJson()).toList(),
};
