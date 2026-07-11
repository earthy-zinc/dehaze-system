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
      if (instance.status case final value?) 'status': value,
    };

DatasetItemModel _$DatasetItemModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('DatasetItemModel', json, ($checkedConvert) {
      final val = DatasetItemModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String?),
        description: $checkedConvert('description', (v) => v as String?),
        files: $checkedConvert(
          'files',
          (v) =>
              (v as List<dynamic>?)
                  ?.map(
                    (e) => ItemFileModel.fromJson(e as Map<String, dynamic>),
                  )
                  .toList() ??
              const [],
        ),
      );
      return val;
    });

Map<String, dynamic> _$DatasetItemModelToJson(DatasetItemModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'datasetId': instance.datasetId,
      if (instance.name case final value?) 'name': value,
      if (instance.description case final value?) 'description': value,
      'files': instance.files.map((e) => e.toJson()).toList(),
      'createTime': instance.createTime,
    };

ItemFileModel _$ItemFileModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ItemFileModel', json, ($checkedConvert) {
      final val = ItemFileModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        itemId: $checkedConvert('itemId', (v) => (v as num).toInt()),
        fileType: $checkedConvert('fileType', (v) => v as String),
        fileUrl: $checkedConvert('fileUrl', (v) => v as String),
        fileId: $checkedConvert('fileId', (v) => v as String?),
        fileName: $checkedConvert('fileName', (v) => v as String?),
        fileSize: $checkedConvert('fileSize', (v) => (v as num?)?.toInt()),
        width: $checkedConvert('width', (v) => (v as num?)?.toInt()),
        height: $checkedConvert('height', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$ItemFileModelToJson(ItemFileModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'itemId': instance.itemId,
      if (instance.fileId case final value?) 'fileId': value,
      'fileType': instance.fileType,
      'fileUrl': instance.fileUrl,
      if (instance.fileName case final value?) 'fileName': value,
      if (instance.fileSize case final value?) 'fileSize': value,
      if (instance.width case final value?) 'width': value,
      if (instance.height case final value?) 'height': value,
    };

ImageModel _$ImageModelFromJson(Map<String, dynamic> json) => $checkedCreate(
  'ImageModel',
  json,
  ($checkedConvert) {
    final val = ImageModel(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      datasetId: $checkedConvert('datasetId', (v) => (v as num).toInt()),
      filename: $checkedConvert('filename', (v) => v as String),
      imageUrl: $checkedConvert('fileUrl', (v) => v as String),
      imageType: $checkedConvert(
        'fileType',
        (v) => $enumDecode(_$ImageTypeEnumMap, v),
      ),
      createdAt: $checkedConvert('createTime', (v) => v as String),
      width: $checkedConvert('width', (v) => (v as num?)?.toInt()),
      height: $checkedConvert('height', (v) => (v as num?)?.toInt()),
      fileSize: $checkedConvert('file_size', (v) => (v as num?)?.toInt()),
      tags: $checkedConvert('tags', (v) => v as String?),
      description: $checkedConvert('description', (v) => v as String?),
    );
    return val;
  },
  fieldKeyMap: const {
    'imageUrl': 'fileUrl',
    'imageType': 'fileType',
    'createdAt': 'createTime',
    'fileSize': 'file_size',
  },
);

Map<String, dynamic> _$ImageModelToJson(ImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'datasetId': instance.datasetId,
      'filename': instance.filename,
      'fileUrl': instance.imageUrl,
      'fileType': _$ImageTypeEnumMap[instance.imageType]!,
      if (instance.width case final value?) 'width': value,
      if (instance.height case final value?) 'height': value,
      if (instance.fileSize case final value?) 'file_size': value,
      if (instance.tags case final value?) 'tags': value,
      if (instance.description case final value?) 'description': value,
      'createTime': instance.createdAt,
    };

const _$ImageTypeEnumMap = {
  ImageType.hazy: 'hazy',
  ImageType.clear: 'clear',
  ImageType.dehazed: 'dehazed',
};
