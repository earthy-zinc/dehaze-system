// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dataset_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DatasetModel _$DatasetModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'DatasetModel',
      json,
      ($checkedConvert) {
        final val = DatasetModel(
          id: $checkedConvert('id', (v) => (v as num).toInt()),
          name: $checkedConvert('name', (v) => v as String),
          description: $checkedConvert('description', (v) => v as String?),
          creator: $checkedConvert('creator', (v) => v as String),
          thumbnail: $checkedConvert('thumbnail', (v) => v as String),
          totalImages: $checkedConvert(
            'total_images',
            (v) => (v as num).toInt(),
          ),
          foggyCount: $checkedConvert('foggy_count', (v) => (v as num).toInt()),
          clearCount: $checkedConvert('clear_count', (v) => (v as num).toInt()),
          annotatedCount: $checkedConvert(
            'annotated_count',
            (v) => (v as num).toInt(),
          ),
          createdAt: $checkedConvert(
            'created_at',
            (v) => DateTime.parse(v as String),
          ),
          updatedAt: $checkedConvert(
            'updated_at',
            (v) => DateTime.parse(v as String),
          ),
        );
        return val;
      },
      fieldKeyMap: const {
        'totalImages': 'total_images',
        'foggyCount': 'foggy_count',
        'clearCount': 'clear_count',
        'annotatedCount': 'annotated_count',
        'createdAt': 'created_at',
        'updatedAt': 'updated_at',
      },
    );

Map<String, dynamic> _$DatasetModelToJson(DatasetModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      if (instance.description case final value?) 'description': value,
      'creator': instance.creator,
      'thumbnail': instance.thumbnail,
      'total_images': instance.totalImages,
      'foggy_count': instance.foggyCount,
      'clear_count': instance.clearCount,
      'annotated_count': instance.annotatedCount,
      'created_at': instance.createdAt.toIso8601String(),
      'updated_at': instance.updatedAt.toIso8601String(),
    };

PaginatedDatasetResponse _$PaginatedDatasetResponseFromJson(
  Map<String, dynamic> json,
) => $checkedCreate(
  'PaginatedDatasetResponse',
  json,
  ($checkedConvert) {
    final val = PaginatedDatasetResponse(
      list: $checkedConvert(
        'list',
        (v) => (v as List<dynamic>)
            .map((e) => DatasetModel.fromJson(e as Map<String, dynamic>))
            .toList(),
      ),
      total: $checkedConvert('total', (v) => (v as num).toInt()),
      page: $checkedConvert('page', (v) => (v as num).toInt()),
      pageSize: $checkedConvert('page_size', (v) => (v as num).toInt()),
      totalPages: $checkedConvert('total_pages', (v) => (v as num).toInt()),
    );
    return val;
  },
  fieldKeyMap: const {'pageSize': 'page_size', 'totalPages': 'total_pages'},
);

Map<String, dynamic> _$PaginatedDatasetResponseToJson(
  PaginatedDatasetResponse instance,
) => <String, dynamic>{
  'list': instance.list.map((e) => e.toJson()).toList(),
  'total': instance.total,
  'page': instance.page,
  'page_size': instance.pageSize,
  'total_pages': instance.totalPages,
};

ImageModel _$ImageModelFromJson(Map<String, dynamic> json) => $checkedCreate(
  'ImageModel',
  json,
  ($checkedConvert) {
    final val = ImageModel(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      datasetId: $checkedConvert('dataset_id', (v) => (v as num).toInt()),
      filename: $checkedConvert('filename', (v) => v as String),
      imageUrl: $checkedConvert('image_url', (v) => v as String),
      imageType: $checkedConvert(
        'image_type',
        (v) => $enumDecode(_$ImageTypeEnumMap, v),
      ),
      width: $checkedConvert('width', (v) => (v as num).toInt()),
      height: $checkedConvert('height', (v) => (v as num).toInt()),
      fileSize: $checkedConvert('file_size', (v) => (v as num?)?.toInt()),
      tags: $checkedConvert('tags', (v) => v as String?),
      description: $checkedConvert('description', (v) => v as String?),
      createdAt: $checkedConvert(
        'created_at',
        (v) => DateTime.parse(v as String),
      ),
    );
    return val;
  },
  fieldKeyMap: const {
    'datasetId': 'dataset_id',
    'imageUrl': 'image_url',
    'imageType': 'image_type',
    'fileSize': 'file_size',
    'createdAt': 'created_at',
  },
);

Map<String, dynamic> _$ImageModelToJson(ImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'dataset_id': instance.datasetId,
      'filename': instance.filename,
      'image_url': instance.imageUrl,
      'image_type': _$ImageTypeEnumMap[instance.imageType]!,
      'width': instance.width,
      'height': instance.height,
      if (instance.fileSize case final value?) 'file_size': value,
      if (instance.tags case final value?) 'tags': value,
      if (instance.description case final value?) 'description': value,
      'created_at': instance.createdAt.toIso8601String(),
    };

const _$ImageTypeEnumMap = {
  ImageType.foggy: 'foggy',
  ImageType.clear: 'clear',
  ImageType.annotated: 'annotated',
};

PaginatedImageResponse _$PaginatedImageResponseFromJson(
  Map<String, dynamic> json,
) => $checkedCreate(
  'PaginatedImageResponse',
  json,
  ($checkedConvert) {
    final val = PaginatedImageResponse(
      list: $checkedConvert(
        'list',
        (v) => (v as List<dynamic>)
            .map((e) => ImageModel.fromJson(e as Map<String, dynamic>))
            .toList(),
      ),
      total: $checkedConvert('total', (v) => (v as num).toInt()),
      page: $checkedConvert('page', (v) => (v as num).toInt()),
      pageSize: $checkedConvert('page_size', (v) => (v as num).toInt()),
      totalPages: $checkedConvert('total_pages', (v) => (v as num).toInt()),
    );
    return val;
  },
  fieldKeyMap: const {'pageSize': 'page_size', 'totalPages': 'total_pages'},
);

Map<String, dynamic> _$PaginatedImageResponseToJson(
  PaginatedImageResponse instance,
) => <String, dynamic>{
  'list': instance.list.map((e) => e.toJson()).toList(),
  'total': instance.total,
  'page': instance.page,
  'page_size': instance.pageSize,
  'total_pages': instance.totalPages,
};
