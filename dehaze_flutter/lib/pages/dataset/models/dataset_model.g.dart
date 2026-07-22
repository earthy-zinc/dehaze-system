// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dataset_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DatasetModel _$DatasetModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetModel', json, ($checkedConvert) {
      final val = DatasetModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        parentId: $checkedConvert('parentId', (v) => (v as num?)?.toInt()),
        type: $checkedConvert('type', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        remark: $checkedConvert('remark', (v) => v as String?),
        usageCount: $checkedConvert('usageCount', (v) => (v as num?)?.toInt()),
        createBy: $checkedConvert('createBy', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
        updateBy: $checkedConvert('updateBy', (v) => v as String?),
        children: $checkedConvert(
          'children',
          (v) =>
              (v as List<dynamic>?)
                  ?.map((e) => DatasetModel.fromJson(e as Map<String, dynamic>))
                  .toList() ??
              const [],
        ),
        hasChildren: $checkedConvert('hasChildren', (v) => v as bool? ?? false),
        total: $checkedConvert('total', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$DatasetModelToJson(DatasetModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.parentId case final value?) 'parentId': value,
      'name': instance.name,
      if (instance.type case final value?) 'type': value,
      if (instance.path case final value?) 'path': value,
      if (instance.description case final value?) 'description': value,
      if (instance.remark case final value?) 'remark': value,
      if (instance.usageCount case final value?) 'usageCount': value,
      if (instance.createBy case final value?) 'createBy': value,
      'createTime': instance.createTime,
      if (instance.updateBy case final value?) 'updateBy': value,
      if (instance.updateTime case final value?) 'updateTime': value,
      'children': instance.children.map((e) => e.toJson()).toList(),
      'hasChildren': instance.hasChildren,
      if (instance.total case final value?) 'total': value,
      if (instance.status case final value?) 'status': value,
    };

DatasetItemModel _$DatasetItemModelFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('DatasetItemModel', json, ($checkedConvert) {
  final val = DatasetItemModel(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
    name: $checkedConvert('name', (v) => v as String?),
    sceneType: $checkedConvert('sceneType', (v) => v as String?),
    description: $checkedConvert('description', (v) => v as String?),
    usageCount: $checkedConvert('usageCount', (v) => (v as num?)?.toInt()),
    imageCount: $checkedConvert('imageCount', (v) => (v as num?)?.toInt()),
    clearImage: $checkedConvert(
      'clearImage',
      (v) =>
          v == null ? null : ItemImageModel.fromJson(v as Map<String, dynamic>),
    ),
    hazyImages: $checkedConvert(
      'hazyImages',
      (v) =>
          (v as List<dynamic>?)
              ?.map((e) => ItemImageModel.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
    ),
    updateTime: $checkedConvert('updateTime', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$DatasetItemModelToJson(DatasetItemModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'datasetId': instance.datasetId,
      if (instance.name case final value?) 'name': value,
      if (instance.sceneType case final value?) 'sceneType': value,
      if (instance.description case final value?) 'description': value,
      if (instance.usageCount case final value?) 'usageCount': value,
      if (instance.imageCount case final value?) 'imageCount': value,
      if (instance.clearImage?.toJson() case final value?) 'clearImage': value,
      'hazyImages': instance.hazyImages.map((e) => e.toJson()).toList(),
      'createTime': instance.createTime,
      if (instance.updateTime case final value?) 'updateTime': value,
    };

ItemImageModel _$ItemImageModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ItemImageModel', json, ($checkedConvert) {
      final val = ItemImageModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        url: $checkedConvert('url', (v) => v as String),
        itemId: $checkedConvert('itemId', (v) => (v as num?)?.toInt()),
        datasetId: $checkedConvert('datasetId', (v) => (v as num?)?.toInt()),
        type: $checkedConvert('type', (v) => v as String?),
        originUrl: $checkedConvert('originUrl', (v) => v as String?),
        thumbnailUrl: $checkedConvert('thumbnailUrl', (v) => v as String?),
        fileName: $checkedConvert('fileName', (v) => v as String?),
        width: $checkedConvert('width', (v) => (v as num?)?.toInt()),
        height: $checkedConvert('height', (v) => (v as num?)?.toInt()),
        sizeBytes: $checkedConvert('sizeBytes', (v) => (v as num?)?.toInt()),
        hazeLevel: $checkedConvert('hazeLevel', (v) => v as String?),
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ItemImageModelToJson(ItemImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      if (instance.itemId case final value?) 'itemId': value,
      if (instance.datasetId case final value?) 'datasetId': value,
      if (instance.type case final value?) 'type': value,
      'url': instance.url,
      if (instance.originUrl case final value?) 'originUrl': value,
      if (instance.thumbnailUrl case final value?) 'thumbnailUrl': value,
      if (instance.fileName case final value?) 'fileName': value,
      if (instance.width case final value?) 'width': value,
      if (instance.height case final value?) 'height': value,
      if (instance.sizeBytes case final value?) 'sizeBytes': value,
      if (instance.hazeLevel case final value?) 'hazeLevel': value,
      if (instance.sceneType case final value?) 'sceneType': value,
      if (instance.description case final value?) 'description': value,
      if (instance.createTime case final value?) 'createTime': value,
    };
