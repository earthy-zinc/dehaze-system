import 'package:json_annotation/json_annotation.dart';

part 'favorite_model.g.dart';

/// 收藏目标类型枚举
///
/// 对应后端 FavoriteTargetType，用于标识被收藏的业务实体类型。
enum FavoriteTargetType {
  @JsonValue('model')
  model,

  @JsonValue('result')
  result,

  @JsonValue('dataset')
  dataset,

  @JsonValue('image')
  image,

  @JsonValue('preset')
  preset,
}

/// 收藏目标类型的线上取值（与后端字符串枚举一致）
extension FavoriteTargetTypeWire on FavoriteTargetType {
  String get wireValue => _$FavoriteTargetTypeEnumMap[this]!;
}

/// 收藏排序字段枚举
///
/// 取值对齐后端 FavoritePageQuery.sortBy（createTime/rating/usageCount）。
@JsonEnum(alwaysCreate: true)
enum FavoriteSortBy {
  @JsonValue('createTime')
  createTime,

  @JsonValue('rating')
  rating,

  @JsonValue('usageCount')
  usageCount,
}

/// 收藏查询参数
///
/// 对应 JS SDK FavoriteQuery，用于分页查询收藏列表。
class FavoriteQuery {
  const FavoriteQuery({
    this.targetType,
    this.keywords,
    this.sortBy,
    this.sortOrder,
    this.pageNum = 1,
    this.pageSize = 20,
  });

  /// 收藏对象类型筛选，不传表示全部类型
  final FavoriteTargetType? targetType;

  /// 关键词搜索（按收藏对象名称）
  final String? keywords;

  /// 排序字段
  final FavoriteSortBy? sortBy;

  /// 排序方向（asc/desc）
  final String? sortOrder;

  final int pageNum;
  final int pageSize;

  Map<String, dynamic> toQuery() => {
    if (targetType != null)
      'targetType': _$FavoriteTargetTypeEnumMap[targetType!],
    if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
    if (sortBy != null) 'sortBy': _$FavoriteSortByEnumMap[sortBy!],
    if (sortOrder != null) 'sortOrder': sortOrder,
    'pageNum': pageNum,
    'pageSize': pageSize,
  };
}

/// 收藏表单
///
/// 对应 JS SDK FavoriteForm，用于添加收藏。
@JsonSerializable()
class FavoriteForm {
  const FavoriteForm({required this.targetType, required this.targetId});

  factory FavoriteForm.fromJson(Map<String, dynamic> json) =>
      _$FavoriteFormFromJson(json);

  final FavoriteTargetType targetType;
  final int targetId;

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
    required this.createTime,
    this.targetName,
    this.targetSummary,
    this.targetThumbnail,
    this.isInvalid = false,
  });

  factory FavoriteVO.fromJson(Map<String, dynamic> json) =>
      _$FavoriteVOFromJson(json);

  final int id;
  final int userId;
  final String targetType;
  final int targetId;

  /// 收藏对象名称（关联查询）
  final String? targetName;

  /// 对象摘要
  final String? targetSummary;

  /// 缩略图 URL
  final String? targetThumbnail;

  /// 是否已失效（对象被删除）
  final bool isInvalid;

  final String createTime;

  Map<String, dynamic> toJson() => _$FavoriteVOToJson(this);
}

/// 收藏状态
///
/// 对应 JS SDK FavoriteStatus，用于前端图标状态判断。
@JsonSerializable()
class FavoriteStatus {
  const FavoriteStatus({
    required this.favorited,
    this.targetType,
    this.targetId,
  });

  factory FavoriteStatus.fromJson(Map<String, dynamic> json) =>
      _$FavoriteStatusFromJson(json);

  /// 收藏对象类型
  final String? targetType;

  /// 收藏对象 ID
  final int? targetId;

  final bool favorited;

  Map<String, dynamic> toJson() => _$FavoriteStatusToJson(this);
}

/// 收藏计数
///
/// 对应 JS SDK FavoriteCount，按类型分组返回，无总数接口。
@JsonSerializable()
class FavoriteCount {
  const FavoriteCount({required this.targetType, required this.count});

  factory FavoriteCount.fromJson(Map<String, dynamic> json) =>
      _$FavoriteCountFromJson(json);

  final String targetType;
  final int count;

  Map<String, dynamic> toJson() => _$FavoriteCountToJson(this);
}
