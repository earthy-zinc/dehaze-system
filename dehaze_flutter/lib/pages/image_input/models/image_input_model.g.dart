// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'image_input_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SelectedImageModel _$SelectedImageModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'SelectedImageModel',
      json,
      ($checkedConvert) {
        final val = SelectedImageModel(
          id: $checkedConvert('id', (v) => v as String),
          url: $checkedConvert('url', (v) => v as String),
          filename: $checkedConvert('filename', (v) => v as String),
          width: $checkedConvert('width', (v) => (v as num).toInt()),
          height: $checkedConvert('height', (v) => (v as num).toInt()),
          fileSize: $checkedConvert('file_size', (v) => (v as num).toInt()),
          source: $checkedConvert(
            'source',
            (v) => $enumDecode(_$ImageSourceEnumMap, v),
          ),
          localPath: $checkedConvert('local_path', (v) => v as String?),
          sampleInfo: $checkedConvert(
            'sample_info',
            (v) => v == null
                ? null
                : SampleImageModel.fromJson(v as Map<String, dynamic>),
          ),
        );
        return val;
      },
      fieldKeyMap: const {
        'fileSize': 'file_size',
        'localPath': 'local_path',
        'sampleInfo': 'sample_info',
      },
    );

Map<String, dynamic> _$SelectedImageModelToJson(SelectedImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'url': instance.url,
      if (instance.localPath case final value?) 'local_path': value,
      'filename': instance.filename,
      'width': instance.width,
      'height': instance.height,
      'file_size': instance.fileSize,
      'source': _$ImageSourceEnumMap[instance.source]!,
      if (instance.sampleInfo?.toJson() case final value?) 'sample_info': value,
    };

const _$ImageSourceEnumMap = {
  ImageSource.upload: 'upload',
  ImageSource.camera: 'camera',
  ImageSource.sample: 'sample',
};

SampleImageModel _$SampleImageModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'SampleImageModel',
      json,
      ($checkedConvert) {
        final val = SampleImageModel(
          id: $checkedConvert('id', (v) => (v as num).toInt()),
          name: $checkedConvert('name', (v) => v as String),
          url: $checkedConvert('url', (v) => v as String),
          category: $checkedConvert(
            'category',
            (v) => $enumDecode(_$SampleCategoryEnumMap, v),
          ),
          difficulty: $checkedConvert(
            'difficulty',
            (v) => $enumDecode(_$DifficultyLevelEnumMap, v),
          ),
          sceneType: $checkedConvert('scene_type', (v) => v as String?),
          recommendedAlgorithm: $checkedConvert(
            'recommended_algorithm',
            (v) => v as String?,
          ),
          cleanUrl: $checkedConvert('cleanUrl', (v) => v as String?),
        );
        return val;
      },
      fieldKeyMap: const {
        'sceneType': 'scene_type',
        'recommendedAlgorithm': 'recommended_algorithm',
      },
    );

Map<String, dynamic> _$SampleImageModelToJson(SampleImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'url': instance.url,
      'category': _$SampleCategoryEnumMap[instance.category]!,
      'difficulty': _$DifficultyLevelEnumMap[instance.difficulty]!,
      if (instance.sceneType case final value?) 'scene_type': value,
      if (instance.recommendedAlgorithm case final value?)
        'recommended_algorithm': value,
      if (instance.cleanUrl case final value?) 'cleanUrl': value,
    };

const _$SampleCategoryEnumMap = {
  SampleCategory.all: 'all',
  SampleCategory.light: 'light',
  SampleCategory.medium: 'medium',
  SampleCategory.heavy: 'heavy',
  SampleCategory.special: 'special',
};

const _$DifficultyLevelEnumMap = {
  DifficultyLevel.easy: 'easy',
  DifficultyLevel.medium: 'medium',
  DifficultyLevel.hard: 'hard',
};

PaginatedSampleResponse _$PaginatedSampleResponseFromJson(
  Map<String, dynamic> json,
) => $checkedCreate(
  'PaginatedSampleResponse',
  json,
  ($checkedConvert) {
    final val = PaginatedSampleResponse(
      list: $checkedConvert(
        'list',
        (v) => (v as List<dynamic>)
            .map((e) => SampleImageModel.fromJson(e as Map<String, dynamic>))
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

Map<String, dynamic> _$PaginatedSampleResponseToJson(
  PaginatedSampleResponse instance,
) => <String, dynamic>{
  'list': instance.list.map((e) => e.toJson()).toList(),
  'total': instance.total,
  'page': instance.page,
  'page_size': instance.pageSize,
  'total_pages': instance.totalPages,
};
