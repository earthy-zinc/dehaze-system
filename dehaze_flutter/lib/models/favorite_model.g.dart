// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'favorite_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

FavoriteForm _$FavoriteFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteForm', json, ($checkedConvert) {
      final val = FavoriteForm(
        targetType: $checkedConvert(
          'targetType',
          (v) => $enumDecode(_$FavoriteTargetTypeEnumMap, v),
        ),
        targetId: $checkedConvert('targetId', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteFormToJson(FavoriteForm instance) =>
    <String, dynamic>{
      'targetType': _$FavoriteTargetTypeEnumMap[instance.targetType]!,
      'targetId': instance.targetId,
    };

const _$FavoriteTargetTypeEnumMap = {
  FavoriteTargetType.model: 'model',
  FavoriteTargetType.result: 'result',
  FavoriteTargetType.dataset: 'dataset',
  FavoriteTargetType.image: 'image',
  FavoriteTargetType.preset: 'preset',
};

FavoriteVO _$FavoriteVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'FavoriteVO',
  json,
  ($checkedConvert) {
    final val = FavoriteVO(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      userId: $checkedConvert('userId', (v) => (v as num).toInt()),
      targetType: $checkedConvert('targetType', (v) => v as String),
      targetId: $checkedConvert('targetId', (v) => (v as num).toInt()),
      createTime: $checkedConvert('createTime', (v) => v as String),
      targetName: $checkedConvert('targetName', (v) => v as String?),
      targetSummary: $checkedConvert('targetSummary', (v) => v as String?),
      targetThumbnail: $checkedConvert('targetThumbnail', (v) => v as String?),
      isInvalid: $checkedConvert('isInvalid', (v) => v as bool? ?? false),
    );
    return val;
  },
);

Map<String, dynamic> _$FavoriteVOToJson(FavoriteVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      'targetType': instance.targetType,
      'targetId': instance.targetId,
      'targetName': ?instance.targetName,
      'targetSummary': ?instance.targetSummary,
      'targetThumbnail': ?instance.targetThumbnail,
      'isInvalid': instance.isInvalid,
      'createTime': instance.createTime,
    };

FavoriteStatus _$FavoriteStatusFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteStatus', json, ($checkedConvert) {
      final val = FavoriteStatus(
        favorited: $checkedConvert('favorited', (v) => v as bool),
        targetType: $checkedConvert('targetType', (v) => v as String?),
        targetId: $checkedConvert('targetId', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteStatusToJson(FavoriteStatus instance) =>
    <String, dynamic>{
      'targetType': ?instance.targetType,
      'targetId': ?instance.targetId,
      'favorited': instance.favorited,
    };

FavoriteCount _$FavoriteCountFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteCount', json, ($checkedConvert) {
      final val = FavoriteCount(
        targetType: $checkedConvert('targetType', (v) => v as String),
        count: $checkedConvert('count', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteCountToJson(FavoriteCount instance) =>
    <String, dynamic>{
      'targetType': instance.targetType,
      'count': instance.count,
    };

const _$FavoriteSortByEnumMap = {
  FavoriteSortBy.createTime: 'createTime',
  FavoriteSortBy.rating: 'rating',
  FavoriteSortBy.usageCount: 'usageCount',
};
