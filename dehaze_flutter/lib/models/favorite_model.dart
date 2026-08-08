import 'package:json_annotation/json_annotation.dart';

part 'favorite_model.g.dart';

/// 收藏目标类型枚举
///
/// 对应后端 FavoriteTargetType，用于标识被收藏的业务实体类型。
enum FavoriteTargetType {
  @JsonValue('algorithm')
  algorithm,

  @JsonValue('dataset')
  dataset,

  @JsonValue('datasetItem')
  datasetItem,

  @JsonValue('predictionLog')
  predictionLog,
}

/// 收藏排序字段枚举
@JsonEnum(alwaysCreate: true)
enum FavoriteSortBy {
  @JsonValue('createTime')
  createTime,

  @JsonValue('name')
  name,

  @JsonValue('usageCount')
  usageCount,
}

/// 收藏查询参数
///
/// 对应 JS SDK FavoriteQuery，用于分页查询收藏列表。
class FavoriteQuery {
  const FavoriteQuery({
    required this.targetType,
    this.targetId,
    this.pageNum = 1,
    this.pageSize = 10,
    this.sortBy,
  });

  final FavoriteTargetType targetType;
  final int? targetId;
  final int pageNum;
  final int pageSize;
  final FavoriteSortBy? sortBy;

  Map<String, dynamic> toQuery() => {
        'targetType': _$FavoriteTargetTypeEnumMap[targetType],
        if (targetId != null) 'targetId': targetId,
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (sortBy != null) 'sortBy': _$FavoriteSortByEnumMap[sortBy],
      };
}

/// 收藏表单
///
/// 对应 JS SDK FavoriteForm，用于添加收藏。
@JsonSerializable()
class FavoriteForm {
  const FavoriteForm({
    required this.targetType,
    required this.targetId,
    this.remark,
  });

  factory FavoriteForm.fromJson(Map<String, dynamic> json) =>
      _$FavoriteFormFromJson(json);

  final FavoriteTargetType targetType;
  final int targetId;
  final String? remark;

  Map<String, dynamic> toJson() => _$FavoriteFormToJson(this);
}

/// 收藏记录视图对象
///
/// 对应 JS SDK FavoriteVO。
@JsonSerializable()
class FavoriteVO {
  const FavoriteVO({
    required this.id,
    required this.userId,
    required this.targetType,
    required this.targetId,
    this.targetName,
    this.targetDescription,
    this.targetImage,
    this.remark,
    required this.createTime,
  });

  factory FavoriteVO.fromJson(Map<String, dynamic> json) =>
      _$FavoriteVOFromJson(json);

  final int id;
  final int userId;

  @JsonKey(name: 'targetType')
  final String targetType;

  final int targetId;
  final String? targetName;
  final String? targetDescription;
  final String? targetImage;
  final String? remark;
  final String createTime;

  Map<String, dynamic> toJson() => _$FavoriteVOToJson(this);
}

/// 收藏状态
///
/// 对应 JS SDK FavoriteStatus，用于前端图标状态判断。
@JsonSerializable()
class FavoriteStatus {
  const FavoriteStatus({required this.favorited});

  factory FavoriteStatus.fromJson(Map<String, dynamic> json) =>
      _$FavoriteStatusFromJson(json);

  final bool favorited;

  Map<String, dynamic> toJson() => _$FavoriteStatusToJson(this);
}

/// 收藏计数
///
/// 对应 JS SDK FavoriteCount。
@JsonSerializable()
class FavoriteCount {
  const FavoriteCount({required this.count});

  factory FavoriteCount.fromJson(Map<String, dynamic> json) =>
      _$FavoriteCountFromJson(json);

  final int count;

  Map<String, dynamic> toJson() => _$FavoriteCountToJson(this);
}
