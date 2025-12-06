// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'image_input_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SelectedImageModel _$SelectedImageModelFromJson(Map<String, dynamic> json) =>
    SelectedImageModel(
      id: json['id'] as String,
      url: json['url'] as String,
      filename: json['filename'] as String,
      width: (json['width'] as num).toInt(),
      height: (json['height'] as num).toInt(),
      fileSize: (json['file_size'] as num).toInt(),
      source: $enumDecode(_$ImageSourceEnumMap, json['source']),
      localPath: json['local_path'] as String?,
      sampleInfo: json['sample_info'] == null
          ? null
          : SampleImageModel.fromJson(
              json['sample_info'] as Map<String, dynamic>,
            ),
    );

Map<String, dynamic> _$SelectedImageModelToJson(SelectedImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'url': instance.url,
      'local_path': instance.localPath,
      'filename': instance.filename,
      'width': instance.width,
      'height': instance.height,
      'file_size': instance.fileSize,
      'source': _$ImageSourceEnumMap[instance.source]!,
      'sample_info': instance.sampleInfo?.toJson(),
    };

const _$ImageSourceEnumMap = {
  ImageSource.upload: 'upload',
  ImageSource.camera: 'camera',
  ImageSource.sample: 'sample',
  ImageSource.history: 'history',
};

SampleImageModel _$SampleImageModelFromJson(Map<String, dynamic> json) =>
    SampleImageModel(
      id: (json['id'] as num).toInt(),
      name: json['name'] as String,
      url: json['url'] as String,
      category: $enumDecode(_$SampleCategoryEnumMap, json['category']),
      difficulty: $enumDecode(_$DifficultyLevelEnumMap, json['difficulty']),
      sceneType: json['scene_type'] as String?,
      recommendedAlgorithm: json['recommended_algorithm'] as String?,
    );

Map<String, dynamic> _$SampleImageModelToJson(SampleImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'url': instance.url,
      'category': _$SampleCategoryEnumMap[instance.category]!,
      'difficulty': _$DifficultyLevelEnumMap[instance.difficulty]!,
      'scene_type': instance.sceneType,
      'recommended_algorithm': instance.recommendedAlgorithm,
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
) =>
    PaginatedSampleResponse(
      list: (json['list'] as List<dynamic>)
          .map((e) => SampleImageModel.fromJson(e as Map<String, dynamic>))
          .toList(),
      total: (json['total'] as num).toInt(),
      page: (json['page'] as num).toInt(),
      pageSize: (json['page_size'] as num).toInt(),
      totalPages: (json['total_pages'] as num).toInt(),
    );

Map<String, dynamic> _$PaginatedSampleResponseToJson(
  PaginatedSampleResponse instance,
) =>
    <String, dynamic>{
      'list': instance.list.map((e) => e.toJson()).toList(),
      'total': instance.total,
      'page': instance.page,
      'page_size': instance.pageSize,
      'total_pages': instance.totalPages,
    };

HistoryRecordModel _$HistoryRecordModelFromJson(Map<String, dynamic> json) =>
    HistoryRecordModel(
      id: json['id'] as String,
      originalThumbnail: json['original_thumbnail'] as String,
      filename: json['filename'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      isSuccess: json['is_success'] as bool,
      resultThumbnail: json['result_thumbnail'] as String?,
      algorithmName: json['algorithm_name'] as String?,
      parameters: json['parameters'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$HistoryRecordModelToJson(HistoryRecordModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'original_thumbnail': instance.originalThumbnail,
      'result_thumbnail': instance.resultThumbnail,
      'filename': instance.filename,
      'timestamp': instance.timestamp.toIso8601String(),
      'algorithm_name': instance.algorithmName,
      'parameters': instance.parameters,
      'is_success': instance.isSuccess,
    };
