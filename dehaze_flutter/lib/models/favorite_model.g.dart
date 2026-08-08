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
        remark: $checkedConvert('remark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteFormToJson(FavoriteForm instance) =>
    <String, dynamic>{
      'targetType': _$FavoriteTargetTypeEnumMap[instance.targetType]!,
      'targetId': instance.targetId,
      if (instance.remark case final value?) 'remark': value,
    };

const _$FavoriteTargetTypeEnumMap = {
  FavoriteTargetType.algorithm: 'algorithm',
  FavoriteTargetType.dataset: 'dataset',
  FavoriteTargetType.datasetItem: 'datasetItem',
  FavoriteTargetType.predictionLog: 'predictionLog',
};

FavoriteVO _$FavoriteVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteVO', json, ($checkedConvert) {
      final val = FavoriteVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        targetType: $checkedConvert('targetType', (v) => v as String),
        targetId: $checkedConvert('targetId', (v) => (v as num).toInt()),
        targetName: $checkedConvert('targetName', (v) => v as String?),
        targetDescription: $checkedConvert(
          'targetDescription',
          (v) => v as String?,
        ),
        targetImage: $checkedConvert('targetImage', (v) => v as String?),
        remark: $checkedConvert('remark', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteVOToJson(
  FavoriteVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'userId': instance.userId,
  'targetType': instance.targetType,
  'targetId': instance.targetId,
  if (instance.targetName case final value?) 'targetName': value,
  if (instance.targetDescription case final value?) 'targetDescription': value,
  if (instance.targetImage case final value?) 'targetImage': value,
  if (instance.remark case final value?) 'remark': value,
  'createTime': instance.createTime,
};

FavoriteStatus _$FavoriteStatusFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteStatus', json, ($checkedConvert) {
      final val = FavoriteStatus(
        favorited: $checkedConvert('favorited', (v) => v as bool),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteStatusToJson(FavoriteStatus instance) =>
    <String, dynamic>{'favorited': instance.favorited};

FavoriteCount _$FavoriteCountFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FavoriteCount', json, ($checkedConvert) {
      final val = FavoriteCount(
        count: $checkedConvert('count', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FavoriteCountToJson(FavoriteCount instance) =>
    <String, dynamic>{'count': instance.count};

const _$FavoriteSortByEnumMap = {
  FavoriteSortBy.createTime: 'createTime',
  FavoriteSortBy.name: 'name',
  FavoriteSortBy.usageCount: 'usageCount',
};
