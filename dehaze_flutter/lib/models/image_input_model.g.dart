// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'image_input_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ImageInputHistoryVO _$ImageInputHistoryVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageInputHistoryVO', json, ($checkedConvert) {
  final val = ImageInputHistoryVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
    originalImageUrl: $checkedConvert('originalImageUrl', (v) => v as String?),
    originalThumbnailUrl: $checkedConvert(
      'originalThumbnailUrl',
      (v) => v as String?,
    ),
    resultImageUrl: $checkedConvert('resultImageUrl', (v) => v as String?),
    resultThumbnailUrl: $checkedConvert(
      'resultThumbnailUrl',
      (v) => v as String?,
    ),
    algorithmId: $checkedConvert('algorithmId', (v) => (v as num?)?.toInt()),
    algorithmName: $checkedConvert('algorithmName', (v) => v as String?),
    algorithmParams: $checkedConvert('algorithmParams', (v) => v as String?),
    processingTime: $checkedConvert(
      'processingTime',
      (v) => (v as num?)?.toInt(),
    ),
    status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    inputSource: $checkedConvert('inputSource', (v) => v as String?),
    syncStatus: $checkedConvert('syncStatus', (v) => (v as num?)?.toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$ImageInputHistoryVOToJson(
  ImageInputHistoryVO instance,
) => <String, dynamic>{
  'id': instance.id,
  if (instance.userId case final value?) 'userId': value,
  if (instance.originalImageUrl case final value?) 'originalImageUrl': value,
  if (instance.originalThumbnailUrl case final value?)
    'originalThumbnailUrl': value,
  if (instance.resultImageUrl case final value?) 'resultImageUrl': value,
  if (instance.resultThumbnailUrl case final value?)
    'resultThumbnailUrl': value,
  if (instance.algorithmId case final value?) 'algorithmId': value,
  if (instance.algorithmName case final value?) 'algorithmName': value,
  if (instance.algorithmParams case final value?) 'algorithmParams': value,
  if (instance.processingTime case final value?) 'processingTime': value,
  if (instance.status case final value?) 'status': value,
  if (instance.inputSource case final value?) 'inputSource': value,
  if (instance.syncStatus case final value?) 'syncStatus': value,
  if (instance.createTime case final value?) 'createTime': value,
};

ImageInputHistoryQuery _$ImageInputHistoryQueryFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageInputHistoryQuery', json, ($checkedConvert) {
  final val = ImageInputHistoryQuery(
    pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
    pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
    status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    inputSource: $checkedConvert('inputSource', (v) => v as String?),
    startDate: $checkedConvert('startDate', (v) => v as String?),
    endDate: $checkedConvert('endDate', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$ImageInputHistoryQueryToJson(
  ImageInputHistoryQuery instance,
) => <String, dynamic>{
  'pageNum': instance.pageNum,
  'pageSize': instance.pageSize,
  if (instance.status case final value?) 'status': value,
  if (instance.inputSource case final value?) 'inputSource': value,
  if (instance.startDate case final value?) 'startDate': value,
  if (instance.endDate case final value?) 'endDate': value,
};

ImageInputHistoryBatchDeleteForm _$ImageInputHistoryBatchDeleteFormFromJson(
  Map<String, dynamic> json,
) =>
    $checkedCreate('ImageInputHistoryBatchDeleteForm', json, ($checkedConvert) {
      final val = ImageInputHistoryBatchDeleteForm(
        ids: $checkedConvert(
          'ids',
          (v) => (v as List<dynamic>).map((e) => (e as num).toInt()).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ImageInputHistoryBatchDeleteFormToJson(
  ImageInputHistoryBatchDeleteForm instance,
) => <String, dynamic>{'ids': instance.ids};

ImageInputHistorySyncItem _$ImageInputHistorySyncItemFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageInputHistorySyncItem', json, ($checkedConvert) {
  final val = ImageInputHistorySyncItem(
    originalImageUrl: $checkedConvert('originalImageUrl', (v) => v as String?),
    originalThumbnailUrl: $checkedConvert(
      'originalThumbnailUrl',
      (v) => v as String?,
    ),
    resultImageUrl: $checkedConvert('resultImageUrl', (v) => v as String?),
    resultThumbnailUrl: $checkedConvert(
      'resultThumbnailUrl',
      (v) => v as String?,
    ),
    algorithmId: $checkedConvert('algorithmId', (v) => (v as num?)?.toInt()),
    algorithmName: $checkedConvert('algorithmName', (v) => v as String?),
    algorithmParams: $checkedConvert('algorithmParams', (v) => v as String?),
    processingTime: $checkedConvert(
      'processingTime',
      (v) => (v as num?)?.toInt(),
    ),
    inputSource: $checkedConvert('inputSource', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$ImageInputHistorySyncItemToJson(
  ImageInputHistorySyncItem instance,
) => <String, dynamic>{
  if (instance.originalImageUrl case final value?) 'originalImageUrl': value,
  if (instance.originalThumbnailUrl case final value?)
    'originalThumbnailUrl': value,
  if (instance.resultImageUrl case final value?) 'resultImageUrl': value,
  if (instance.resultThumbnailUrl case final value?)
    'resultThumbnailUrl': value,
  if (instance.algorithmId case final value?) 'algorithmId': value,
  if (instance.algorithmName case final value?) 'algorithmName': value,
  if (instance.algorithmParams case final value?) 'algorithmParams': value,
  if (instance.processingTime case final value?) 'processingTime': value,
  if (instance.inputSource case final value?) 'inputSource': value,
};

ImageInputHistorySyncForm _$ImageInputHistorySyncFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('ImageInputHistorySyncForm', json, ($checkedConvert) {
  final val = ImageInputHistorySyncForm(
    items: $checkedConvert(
      'items',
      (v) => (v as List<dynamic>)
          .map(
            (e) =>
                ImageInputHistorySyncItem.fromJson(e as Map<String, dynamic>),
          )
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$ImageInputHistorySyncFormToJson(
  ImageInputHistorySyncForm instance,
) => <String, dynamic>{'items': instance.items.map((e) => e.toJson()).toList()};
