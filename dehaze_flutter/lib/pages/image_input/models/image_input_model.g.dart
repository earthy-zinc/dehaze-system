// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'image_input_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

SelectedImageModel _$SelectedImageModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('SelectedImageModel', json, ($checkedConvert) {
      final val = SelectedImageModel(
        id: $checkedConvert('id', (v) => v as String),
        url: $checkedConvert('url', (v) => v as String),
        filename: $checkedConvert('filename', (v) => v as String),
        width: $checkedConvert('width', (v) => (v as num).toInt()),
        height: $checkedConvert('height', (v) => (v as num).toInt()),
        fileSize: $checkedConvert('fileSize', (v) => (v as num).toInt()),
        source: $checkedConvert(
          'source',
          (v) => $enumDecode(_$ImageSourceEnumMap, v),
        ),
        localPath: $checkedConvert('localPath', (v) => v as String?),
        sampleInfo: $checkedConvert(
          'sampleInfo',
          (v) => v == null
              ? null
              : SampleImageModel.fromJson(v as Map<String, dynamic>),
        ),
      );
      return val;
    });

Map<String, dynamic> _$SelectedImageModelToJson(SelectedImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'url': instance.url,
      if (instance.localPath case final value?) 'localPath': value,
      'filename': instance.filename,
      'width': instance.width,
      'height': instance.height,
      'fileSize': instance.fileSize,
      'source': _$ImageSourceEnumMap[instance.source]!,
      if (instance.sampleInfo?.toJson() case final value?) 'sampleInfo': value,
    };

const _$ImageSourceEnumMap = {
  ImageSource.upload: 'upload',
  ImageSource.camera: 'camera',
  ImageSource.sample: 'sample',
};

SampleImageModel _$SampleImageModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('SampleImageModel', json, ($checkedConvert) {
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
        sceneType: $checkedConvert('sceneType', (v) => v as String?),
        recommendedAlgorithm: $checkedConvert(
          'recommendedAlgorithm',
          (v) => v as String?,
        ),
        cleanUrl: $checkedConvert('cleanUrl', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$SampleImageModelToJson(SampleImageModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'url': instance.url,
      'category': _$SampleCategoryEnumMap[instance.category]!,
      'difficulty': _$DifficultyLevelEnumMap[instance.difficulty]!,
      if (instance.sceneType case final value?) 'sceneType': value,
      if (instance.recommendedAlgorithm case final value?)
        'recommendedAlgorithm': value,
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
