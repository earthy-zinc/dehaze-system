import 'package:json_annotation/json_annotation.dart';

part 'image_input_model.g.dart';

/// 图片输入历史记录视图对象
///
/// 对应 JS SDK InputHistoryVO 及后端 InputHistoryVO。
@JsonSerializable()
class ImageInputHistoryVO {
  const ImageInputHistoryVO({
    required this.id,
    this.userId,
    this.originalImageUrl,
    this.originalThumbnailUrl,
    this.resultImageUrl,
    this.resultThumbnailUrl,
    this.algorithmId,
    this.algorithmName,
    this.algorithmParams,
    this.processingTime,
    this.status,
    this.inputSource,
    this.syncStatus,
    this.createTime,
  });

  factory ImageInputHistoryVO.fromJson(Map<String, dynamic> json) =>
      _$ImageInputHistoryVOFromJson(json);

  final int id;
  final int? userId;

  @JsonKey(name: 'originalImageUrl')
  final String? originalImageUrl;

  @JsonKey(name: 'originalThumbnailUrl')
  final String? originalThumbnailUrl;

  @JsonKey(name: 'resultImageUrl')
  final String? resultImageUrl;

  @JsonKey(name: 'resultThumbnailUrl')
  final String? resultThumbnailUrl;

  final int? algorithmId;
  final String? algorithmName;
  final String? algorithmParams;
  final int? processingTime;
  final int? status;

  @JsonKey(name: 'inputSource')
  final String? inputSource;

  final int? syncStatus;
  final String? createTime;

  Map<String, dynamic> toJson() => _$ImageInputHistoryVOToJson(this);
}

/// 历史记录查询参数
///
/// 对应 JS SDK HistoryQuery。
@JsonSerializable()
class ImageInputHistoryQuery {
  const ImageInputHistoryQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.status,
    this.inputSource,
    this.startDate,
    this.endDate,
  });

  factory ImageInputHistoryQuery.fromJson(Map<String, dynamic> json) =>
      _$ImageInputHistoryQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final int? status;

  @JsonKey(name: 'inputSource')
  final String? inputSource;

  final String? startDate;
  final String? endDate;

  Map<String, dynamic> toJson() => _$ImageInputHistoryQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (status != null) 'status': status,
        if (inputSource != null) 'inputSource': inputSource,
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      };
}

/// 批量删除表单
///
/// 对应后端 BatchDeleteForm。
@JsonSerializable()
class ImageInputHistoryBatchDeleteForm {
  const ImageInputHistoryBatchDeleteForm({required this.ids});

  factory ImageInputHistoryBatchDeleteForm.fromJson(Map<String, dynamic> json) =>
      _$ImageInputHistoryBatchDeleteFormFromJson(json);

  final List<int> ids;

  Map<String, dynamic> toJson() => _$ImageInputHistoryBatchDeleteFormToJson(this);
}

/// 历史记录同步项
///
/// 用于同步本地与云端历史记录时的单项数据。
@JsonSerializable()
class ImageInputHistorySyncItem {
  const ImageInputHistorySyncItem({
    required this.originalImageUrl,
    this.originalThumbnailUrl,
    this.resultImageUrl,
    this.resultThumbnailUrl,
    this.algorithmId,
    this.algorithmName,
    this.algorithmParams,
    this.processingTime,
    this.inputSource,
  });

  factory ImageInputHistorySyncItem.fromJson(Map<String, dynamic> json) =>
      _$ImageInputHistorySyncItemFromJson(json);

  @JsonKey(name: 'originalImageUrl')
  final String? originalImageUrl;

  @JsonKey(name: 'originalThumbnailUrl')
  final String? originalThumbnailUrl;

  @JsonKey(name: 'resultImageUrl')
  final String? resultImageUrl;

  @JsonKey(name: 'resultThumbnailUrl')
  final String? resultThumbnailUrl;

  final int? algorithmId;
  final String? algorithmName;
  final String? algorithmParams;
  final int? processingTime;

  @JsonKey(name: 'inputSource')
  final String? inputSource;

  Map<String, dynamic> toJson() => _$ImageInputHistorySyncItemToJson(this);
}

/// 历史记录同步表单
///
/// 用于批量同步本地与云端历史记录。
@JsonSerializable()
class ImageInputHistorySyncForm {
  const ImageInputHistorySyncForm({required this.items});

  factory ImageInputHistorySyncForm.fromJson(Map<String, dynamic> json) =>
      _$ImageInputHistorySyncFormFromJson(json);

  final List<ImageInputHistorySyncItem> items;

  Map<String, dynamic> toJson() => _$ImageInputHistorySyncFormToJson(this);
}
