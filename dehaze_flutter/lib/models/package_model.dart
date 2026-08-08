import 'package:json_annotation/json_annotation.dart';

part 'package_model.g.dart';

// ==================== 枚举 ====================

/// 套餐等级编码
enum PackageLevelCode {
  @JsonValue('bronze')
  bronze,
  @JsonValue('silver')
  silver,
  @JsonValue('gold')
  gold,
  @JsonValue('platinum')
  platinum,
  @JsonValue('diamond')
  diamond,
}

/// 套餐计费周期
enum PackagePeriod {
  @JsonValue(1)
  monthly,
  @JsonValue(3)
  quarterly,
  @JsonValue(12)
  yearly,
  @JsonValue(0)
  lifetime,
}

/// 套餐状态
enum PackageStatus {
  @JsonValue(0)
  draft,
  @JsonValue(1)
  active,
  @JsonValue(2)
  inactive,
}

/// 优惠券类型
enum CouponType {
  @JsonValue('fixed')
  fixed,
  @JsonValue('percentage')
  percentage,
}

/// 优惠券有效期类型
enum CouponValidType {
  @JsonValue('days')
  days,
  @JsonValue('dateRange')
  dateRange,
}

/// 用户优惠券状态
enum UserCouponStatus {
  @JsonValue(0)
  unused,
  @JsonValue(1)
  used,
  @JsonValue(2)
  expired,
}

// ==================== 权益 VO ====================

/// 权益配置 VO（与 member_model.dart 中 BenefitVO 一致）
@JsonSerializable()
class BenefitVO {
  const BenefitVO({
    required this.levelCode,
    required this.levelName,
    required this.growthMin,
    required this.growthMax,
    required this.monthlyDehazeQuota,
    required this.monthlyEvaluateQuota,
    required this.historyRetention,
    required this.batchLimit,
    required this.priority,
    required this.advancedParams,
    required this.hdExport,
    required this.reportExport,
    required this.batchDownload,
    required this.sort,
    required this.status,
  });

  factory BenefitVO.fromJson(Map<String, dynamic> json) =>
      _$BenefitVOFromJson(json);

  final String levelCode;
  final String levelName;
  final int growthMin;
  final int growthMax;
  final int monthlyDehazeQuota;
  final int monthlyEvaluateQuota;
  final int historyRetention;
  final int batchLimit;
  final int priority;
  final int advancedParams;
  final int hdExport;
  final int reportExport;
  final int batchDownload;
  final int sort;
  final int status;

  Map<String, dynamic> toJson() => _$BenefitVOToJson(this);
}

// ==================== 套餐 ====================

/// 套餐表单
@JsonSerializable()
class PackageForm {
  const PackageForm({
    this.id,
    required this.name,
    required this.level,
    required this.period,
    required this.originalPrice,
    required this.currentPrice,
    this.description,
    this.features = const [],
    this.status,
    this.sort,
    this.benefits,
  });

  factory PackageForm.fromJson(Map<String, dynamic> json) =>
      _$PackageFormFromJson(json);

  final int? id;
  final String name;
  final String level;
  final String period;
  final double originalPrice;
  final double currentPrice;
  final String? description;
  final List<String> features;
  final int? status;
  final int? sort;

  /// JSON 字段，权益覆盖配置
  final Map<String, dynamic>? benefits;

  Map<String, dynamic> toJson() => _$PackageFormToJson(this);
}

/// 套餐分页项
@JsonSerializable()
class PackagePageVO {
  const PackagePageVO({
    required this.id,
    required this.name,
    required this.level,
    required this.levelName,
    required this.period,
    required this.periodName,
    required this.originalPrice,
    required this.currentPrice,
    this.description,
    this.features = const [],
    required this.status,
    this.sort,
    required this.createTime,
  });

  factory PackagePageVO.fromJson(Map<String, dynamic> json) =>
      _$PackagePageVOFromJson(json);

  final int id;
  final String name;
  final String level;
  final String levelName;
  final String period;
  final String periodName;
  final double originalPrice;
  final double currentPrice;
  final String? description;
  final List<String> features;
  final int status;
  final int? sort;
  final String createTime;

  Map<String, dynamic> toJson() => _$PackagePageVOToJson(this);
}

/// 套餐详情
@JsonSerializable()
class PackageDetailVO {
  const PackageDetailVO({
    required this.id,
    required this.name,
    required this.level,
    required this.levelName,
    required this.period,
    required this.periodName,
    required this.originalPrice,
    required this.currentPrice,
    this.description,
    this.features = const [],
    required this.status,
    this.sort,
    required this.createTime,
    this.benefits,
  });

  factory PackageDetailVO.fromJson(Map<String, dynamic> json) =>
      _$PackageDetailVOFromJson(json);

  final int id;
  final String name;
  final String level;
  final String levelName;
  final String period;
  final String periodName;
  final double originalPrice;
  final double currentPrice;
  final String? description;
  final List<String> features;
  final int status;
  final int? sort;
  final String createTime;
  final List<BenefitVO>? benefits;

  Map<String, dynamic> toJson() => _$PackageDetailVOToJson(this);
}

/// 套餐查询参数
@JsonSerializable()
class PackageQuery {
  const PackageQuery({
    this.pageNum,
    this.pageSize,
    this.level,
    this.status,
    this.keyword,
  });

  factory PackageQuery.fromJson(Map<String, dynamic> json) =>
      _$PackageQueryFromJson(json);

  final int? pageNum;
  final int? pageSize;
  final String? level;
  final int? status;
  final String? keyword;

  Map<String, dynamic> toJson() => _$PackageQueryToJson(this);

  /// 转为查询参数（过滤 null 值）
  Map<String, dynamic> toQueryParameters() {
    final map = <String, dynamic>{};
    if (pageNum != null) map['pageNum'] = pageNum;
    if (pageSize != null) map['pageSize'] = pageSize;
    if (level != null) map['level'] = level;
    if (status != null) map['status'] = status;
    if (keyword != null) map['keyword'] = keyword;
    return map;
  }
}

/// 价格计算结果
@JsonSerializable()
class PriceResult {
  const PriceResult({
    required this.originalPrice,
    required this.currentPrice,
    required this.discount,
    required this.couponDiscount,
    required this.finalPrice,
    this.appliedCoupons,
  });

  factory PriceResult.fromJson(Map<String, dynamic> json) =>
      _$PriceResultFromJson(json);

  final double originalPrice;
  final double currentPrice;
  final double discount;
  final double couponDiscount;
  final double finalPrice;
  final List<UserCouponVO>? appliedCoupons;

  Map<String, dynamic> toJson() => _$PriceResultToJson(this);
}

/// 促销信息
@JsonSerializable()
class PromotionVO {
  const PromotionVO({
    required this.title,
    this.description,
    required this.startDate,
    required this.endDate,
    required this.discount,
    required this.active,
  });

  factory PromotionVO.fromJson(Map<String, dynamic> json) =>
      _$PromotionVOFromJson(json);

  final String title;
  final String? description;
  final String startDate;
  final String endDate;
  final double discount;
  final bool active;

  Map<String, dynamic> toJson() => _$PromotionVOToJson(this);
}

// ==================== 优惠券 ====================

/// 优惠券表单
@JsonSerializable()
class CouponForm {
  const CouponForm({
    this.code,
    required this.name,
    required this.type,
    required this.value,
    required this.validType,
    this.validDays,
    this.validStart,
    this.validEnd,
    this.minAmount,
    this.maxDiscount,
    required this.total,
    required this.perUser,
    this.description,
    this.status,
    this.scope,
    this.packageIds,
  });

  factory CouponForm.fromJson(Map<String, dynamic> json) =>
      _$CouponFormFromJson(json);

  final String? code;
  final String name;
  final String type;
  final double value;
  final String validType;
  final int? validDays;
  final String? validStart;
  final String? validEnd;
  final double? minAmount;
  final double? maxDiscount;
  final int total;
  final int perUser;
  final String? description;
  final int? status;
  final int? scope;
  final List<int>? packageIds;

  Map<String, dynamic> toJson() => _$CouponFormToJson(this);
}

/// 优惠券查询参数
@JsonSerializable()
class CouponQuery {
  const CouponQuery({
    this.pageNum,
    this.pageSize,
    this.type,
    this.status,
    this.keyword,
  });

  factory CouponQuery.fromJson(Map<String, dynamic> json) =>
      _$CouponQueryFromJson(json);

  final int? pageNum;
  final int? pageSize;
  final String? type;
  final int? status;
  final String? keyword;

  Map<String, dynamic> toJson() => _$CouponQueryToJson(this);

  Map<String, dynamic> toQueryParameters() {
    final map = <String, dynamic>{};
    if (pageNum != null) map['pageNum'] = pageNum;
    if (pageSize != null) map['pageSize'] = pageSize;
    if (type != null) map['type'] = type;
    if (status != null) map['status'] = status;
    if (keyword != null) map['keyword'] = keyword;
    return map;
  }
}

/// 优惠券 VO
@JsonSerializable()
class CouponVO {
  const CouponVO({
    required this.id,
    required this.code,
    required this.name,
    required this.type,
    required this.typeName,
    required this.value,
    required this.validType,
    this.validDays,
    this.validStart,
    this.validEnd,
    this.minAmount,
    this.maxDiscount,
    required this.total,
    required this.issued,
    required this.remaining,
    required this.perUser,
    this.description,
    required this.status,
    this.scope,
    this.packageIds,
    required this.createTime,
  });

  factory CouponVO.fromJson(Map<String, dynamic> json) =>
      _$CouponVOFromJson(json);

  final int id;
  final String code;
  final String name;
  final String type;
  final String typeName;
  final double value;
  final String validType;
  final int? validDays;
  final String? validStart;
  final String? validEnd;
  final double? minAmount;
  final double? maxDiscount;
  final int total;
  final int issued;
  final int remaining;
  final int perUser;
  final String? description;
  final int status;
  final int? scope;
  final List<int>? packageIds;
  final String createTime;

  Map<String, dynamic> toJson() => _$CouponVOToJson(this);
}

/// 用户优惠券 VO
@JsonSerializable()
class UserCouponVO {
  const UserCouponVO({
    required this.id,
    required this.couponId,
    required this.couponName,
    required this.couponType,
    required this.couponValue,
    required this.status,
    required this.validStart,
    required this.validEnd,
    this.minAmount,
    this.maxDiscount,
    this.usedTime,
    this.orderId,
    required this.createTime,
  });

  factory UserCouponVO.fromJson(Map<String, dynamic> json) =>
      _$UserCouponVOFromJson(json);

  final int id;
  final int couponId;
  final String couponName;
  final String couponType;
  final double couponValue;
  final int status;
  final String validStart;
  final String validEnd;
  final double? minAmount;
  final double? maxDiscount;
  final String? usedTime;
  final int? orderId;
  final String createTime;

  Map<String, dynamic> toJson() => _$UserCouponVOToJson(this);
}

/// 优惠券批量发放表单
@JsonSerializable()
class CouponBatchDistributeForm {
  const CouponBatchDistributeForm({
    required this.couponId,
    required this.userIds,
    required this.count,
  });

  factory CouponBatchDistributeForm.fromJson(Map<String, dynamic> json) =>
      _$CouponBatchDistributeFormFromJson(json);

  final int couponId;
  final List<int> userIds;
  final int count;

  Map<String, dynamic> toJson() => _$CouponBatchDistributeFormToJson(this);
}

// ==================== 销售统计 ====================

/// 月收入
@JsonSerializable()
class MonthlyRevenue {
  const MonthlyRevenue({
    required this.month,
    required this.revenue,
    required this.orders,
  });

  factory MonthlyRevenue.fromJson(Map<String, dynamic> json) =>
      _$MonthlyRevenueFromJson(json);

  final String month;
  final double revenue;
  final int orders;

  Map<String, dynamic> toJson() => _$MonthlyRevenueToJson(this);
}

/// 套餐销售
@JsonSerializable()
class PackageSales {
  const PackageSales({
    required this.packageId,
    required this.packageName,
    required this.sales,
    required this.revenue,
  });

  factory PackageSales.fromJson(Map<String, dynamic> json) =>
      _$PackageSalesFromJson(json);

  final int packageId;
  final String packageName;
  final int sales;
  final double revenue;

  Map<String, dynamic> toJson() => _$PackageSalesToJson(this);
}

/// 销售统计
@JsonSerializable()
class SalesStatsVO {
  const SalesStatsVO({
    required this.totalSales,
    required this.totalOrders,
    required this.activeSubscriptions,
    required this.renewalRate,
    this.monthlyRevenue = const [],
    this.topPackages = const [],
  });

  factory SalesStatsVO.fromJson(Map<String, dynamic> json) =>
      _$SalesStatsVOFromJson(json);

  final double totalSales;
  final int totalOrders;
  final int activeSubscriptions;
  final double renewalRate;
  final List<MonthlyRevenue> monthlyRevenue;
  final List<PackageSales> topPackages;

  Map<String, dynamic> toJson() => _$SalesStatsVOToJson(this);
}
