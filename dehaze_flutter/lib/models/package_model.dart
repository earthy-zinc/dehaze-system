import 'package:json_annotation/json_annotation.dart';

part 'package_model.g.dart';

// ==================== 枚举 ====================

/// 优惠券类型
enum CouponType {
  @JsonValue('full_reduction')
  fullReduction,
  @JsonValue('discount')
  discount,
  @JsonValue('no_threshold')
  noThreshold,
  @JsonValue('trial')
  trial,
}

/// 优惠券有效期类型
enum CouponValidType {
  @JsonValue('fixed')
  fixed,
  @JsonValue('relative')
  relative,
}

/// 用户优惠券状态（1 未使用 / 2 已使用 / 3 已过期 / 4 已锁定）
enum UserCouponStatus {
  unused(1, '未使用'),
  used(2, '已使用'),
  expired(3, '已过期'),
  locked(4, '已锁定');

  const UserCouponStatus(this.value, this.displayName);

  final int value;
  final String displayName;
}

// ==================== JSON 转换辅助函数 ====================

UserCouponStatus _userCouponStatusFromJson(dynamic value) {
  final code = value is num ? value.toInt() : null;
  return UserCouponStatus.values.firstWhere(
    (e) => e.value == code,
    orElse: () => UserCouponStatus.unused,
  );
}

int _userCouponStatusToJson(UserCouponStatus status) => status.value;

// ==================== 套餐 ====================

/// 计费周期中文标签（与 Vue package/shop 一致）
const Map<String, String> packagePeriodLabels = {
  'monthly': '月卡',
  'quarterly': '季卡',
  'yearly': '年卡',
};

/// 权益字段中文标签（与 Vue package/shop 一致）
const Map<String, String> packageBenefitLabels = {
  'monthlyDehazeQuota': '去雾配额',
  'monthlyDerainQuota': '去雨配额',
  'monthlyDesnowQuota': '去雪配额',
  'monthlyLowlightQuota': '低光增强配额',
  'monthlySuperResolutionQuota': '超分辨率配额',
  'monthlyDenoiseQuota': '去噪配额',
  'monthlyInpaintQuota': '图像修复配额',
  'monthlyEvaluateQuota': '评估配额',
  'aiCreditsDaily': 'AI对话日限额',
  'aiCreditsMonthly': 'AI对话月限额',
  'historyRetention': '历史保留',
  'batchLimit': '批量上限',
  'priority': '优先级',
  'advancedParams': '高级参数',
  'hdExport': '高清导出',
  'reportExport': '报告导出',
  'batchDownload': '批量下载',
};

/// 商品类型中文
String packageTypeLabel(String packageType) =>
    packageType == 'credit' ? '积分卡' : '会员卡';

/// 套餐规格副标题（积分卡展示积分，会员卡展示周期与有效期）
String packageSpecLabel(PackageDetailVO pkg) {
  if (pkg.packageType == 'credit') {
    return '${pkg.creditAmount ?? 0} 积分';
  }
  final period = packagePeriodLabels[pkg.period] ?? pkg.period ?? '';
  return pkg.periodDays != null ? '$period · ${pkg.periodDays} 天' : period;
}

/// 权益覆盖项（套餐表单 benefitOverrides）
@JsonSerializable()
class BenefitOverrides {
  const BenefitOverrides({
    this.monthlyDehazeQuota,
    this.monthlyDerainQuota,
    this.monthlyDesnowQuota,
    this.monthlyLowlightQuota,
    this.monthlySuperResolutionQuota,
    this.monthlyDenoiseQuota,
    this.monthlyInpaintQuota,
    this.monthlyEvaluateQuota,
    this.aiCreditsDaily,
    this.aiCreditsMonthly,
    this.historyRetention,
    this.batchLimit,
    this.priority,
    this.advancedParams,
    this.hdExport,
    this.reportExport,
    this.batchDownload,
  });

  factory BenefitOverrides.fromJson(Map<String, dynamic> json) =>
      _$BenefitOverridesFromJson(json);

  final int? monthlyDehazeQuota;
  final int? monthlyDerainQuota;
  final int? monthlyDesnowQuota;
  final int? monthlyLowlightQuota;
  final int? monthlySuperResolutionQuota;
  final int? monthlyDenoiseQuota;
  final int? monthlyInpaintQuota;
  final int? monthlyEvaluateQuota;
  final int? aiCreditsDaily;
  final int? aiCreditsMonthly;
  final int? historyRetention;
  final int? batchLimit;
  final int? priority;
  final int? advancedParams;
  final int? hdExport;
  final int? reportExport;
  final int? batchDownload;

  Map<String, dynamic> toJson() => _$BenefitOverridesToJson(this);
}

/// 套餐表单（vip 卡需 levelCode/period/periodDays；credit 卡需 creditAmount）
@JsonSerializable()
class PackageForm {
  const PackageForm({
    this.id,
    required this.name,
    required this.packageType,
    required this.originalPrice,
    required this.salePrice,
    this.levelCode,
    this.period,
    this.periodDays,
    this.creditAmount,
    this.description,
    this.benefitOverrides,
    this.sort,
    this.status,
  });

  factory PackageForm.fromJson(Map<String, dynamic> json) =>
      _$PackageFormFromJson(json);

  final int? id;
  final String name;

  /// vip | credit
  final String packageType;
  final String? levelCode;
  final String? period;
  final int? periodDays;
  final int? creditAmount;

  /// 金额单位：分
  final int originalPrice;
  final int salePrice;
  final String? description;
  final BenefitOverrides? benefitOverrides;
  final int? sort;
  final int? status;

  Map<String, dynamic> toJson() => _$PackageFormToJson(this);
}

/// 套餐分页项（积分卡的等级/周期/有效期字段为 null）
@JsonSerializable()
class PackagePageVO {
  const PackagePageVO({
    required this.id,
    required this.name,
    required this.packageType,
    required this.originalPrice,
    required this.salePrice,
    required this.dailyPrice,
    required this.status,
    required this.createTime,
    this.levelCode,
    this.levelName,
    this.period,
    this.periodDays,
    this.creditAmount,
    this.creditUnitPrice,
    this.salesCount,
  });

  factory PackagePageVO.fromJson(Map<String, dynamic> json) =>
      _$PackagePageVOFromJson(json);

  final int id;
  final String name;

  /// vip | credit
  final String packageType;
  final String? levelCode;
  final String? levelName;
  final String? period;
  final int? periodDays;
  final int? creditAmount;

  /// 金额单位：分
  final int originalPrice;
  final int salePrice;
  final int dailyPrice;
  final int? creditUnitPrice;
  final int? salesCount;

  /// 1:上架;0:下架
  final int status;
  final String createTime;

  Map<String, dynamic> toJson() => _$PackagePageVOToJson(this);
}

/// 套餐详情（用户端；列表接口不返回 activePromotions）
@JsonSerializable()
class PackageDetailVO {
  const PackageDetailVO({
    required this.id,
    required this.name,
    required this.packageType,
    required this.originalPrice,
    required this.salePrice,
    required this.dailyPrice,
    required this.benefits,
    required this.salesCount,
    this.levelCode,
    this.levelName,
    this.period,
    this.periodDays,
    this.creditAmount,
    this.creditUnitPrice,
    this.description,
    this.activePromotions,
  });

  factory PackageDetailVO.fromJson(Map<String, dynamic> json) =>
      _$PackageDetailVOFromJson(json);

  final int id;
  final String name;

  /// vip | credit
  final String packageType;
  final String? levelCode;
  final String? levelName;
  final String? period;
  final int? periodDays;

  /// 金额单位：分
  final int originalPrice;
  final int salePrice;
  final int dailyPrice;
  final int? creditAmount;
  final int? creditUnitPrice;
  final String? description;

  /// 实际生效权益，key → 数值
  final Map<String, int> benefits;
  final List<PromotionVO>? activePromotions;
  final int salesCount;

  Map<String, dynamic> toJson() => _$PackageDetailVOToJson(this);
}

/// 套餐查询参数
@JsonSerializable()
class PackageQuery {
  const PackageQuery({
    this.pageNum,
    this.pageSize,
    this.name,
    this.packageType,
    this.levelCode,
    this.period,
    this.status,
  });

  factory PackageQuery.fromJson(Map<String, dynamic> json) =>
      _$PackageQueryFromJson(json);

  final int? pageNum;
  final int? pageSize;
  final String? name;
  final String? packageType;
  final String? levelCode;
  final String? period;
  final int? status;

  Map<String, dynamic> toJson() => _$PackageQueryToJson(this);

  /// 转为查询参数（过滤 null 值）
  Map<String, dynamic> toQueryParameters() {
    final map = <String, dynamic>{};
    if (pageNum != null) map['pageNum'] = pageNum;
    if (pageSize != null) map['pageSize'] = pageSize;
    if (name != null) map['name'] = name;
    if (packageType != null) map['packageType'] = packageType;
    if (levelCode != null) map['levelCode'] = levelCode;
    if (period != null) map['period'] = period;
    if (status != null) map['status'] = status;
    return map;
  }
}

/// 价格计算结果（金额单位：分）
@JsonSerializable()
class PriceResult {
  const PriceResult({
    required this.originalPrice,
    required this.discountAmount,
    required this.couponAmount,
    required this.payableAmount,
  });

  factory PriceResult.fromJson(Map<String, dynamic> json) =>
      _$PriceResultFromJson(json);

  final int originalPrice;
  final int discountAmount;
  final int couponAmount;
  final int payableAmount;

  Map<String, dynamic> toJson() => _$PriceResultToJson(this);
}

/// 促销活动 VO
@JsonSerializable()
class PromotionVO {
  const PromotionVO({
    required this.id,
    required this.name,
    required this.type,
    required this.startTime,
    required this.endTime,
    this.description,
    this.activityRules,
    this.newUserOnly,
    this.status,
  });

  factory PromotionVO.fromJson(Map<String, dynamic> json) =>
      _$PromotionVOFromJson(json);

  final int id;
  final String name;
  final String type;
  final String? description;
  final String startTime;
  final String endTime;
  final Map<String, dynamic>? activityRules;
  final int? newUserOnly;
  final int? status;

  Map<String, dynamic> toJson() => _$PromotionVOToJson(this);
}

// ==================== 优惠券 ====================

/// 优惠券表单
@JsonSerializable()
class CouponForm {
  const CouponForm({
    this.id,
    required this.name,
    required this.type,
    required this.faceValue,
    required this.validType,
    required this.totalQty,
    required this.perUserLimit,
    this.threshold,
    this.validStart,
    this.validEnd,
    this.validDays,
    this.applicableScope,
    this.status,
  });

  factory CouponForm.fromJson(Map<String, dynamic> json) =>
      _$CouponFormFromJson(json);

  final int? id;
  final String name;
  final CouponType type;

  /// 面额（分）
  final int faceValue;

  /// 使用门槛（分），满减类型必填
  final int? threshold;
  final CouponValidType validType;
  final String? validStart;
  final String? validEnd;
  final int? validDays;
  final int totalQty;
  final int perUserLimit;

  /// 适用商品：商品 ID 或商品类型（NULL / 空表示全部适用）
  final List<dynamic>? applicableScope;
  final int? status;

  Map<String, dynamic> toJson() => _$CouponFormToJson(this);
}

/// 优惠券查询参数
@JsonSerializable()
class CouponQuery {
  const CouponQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.name,
    this.type,
    this.status,
  });

  factory CouponQuery.fromJson(Map<String, dynamic> json) =>
      _$CouponQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? name;
  final String? type;
  final int? status;

  Map<String, dynamic> toJson() => _$CouponQueryToJson(this);

  Map<String, dynamic> toQueryParameters() {
    final map = <String, dynamic>{'pageNum': pageNum, 'pageSize': pageSize};
    if (name != null) map['name'] = name;
    if (type != null) map['type'] = type;
    if (status != null) map['status'] = status;
    return map;
  }
}

/// 优惠券模板 VO
@JsonSerializable()
class CouponVO {
  const CouponVO({
    required this.id,
    required this.name,
    required this.type,
    required this.faceValue,
    required this.validType,
    required this.totalQty,
    required this.issuedQty,
    required this.usedQty,
    required this.perUserLimit,
    required this.status,
    required this.createTime,
    this.threshold,
    this.validStart,
    this.validEnd,
    this.validDays,
    this.applicableScope,
  });

  factory CouponVO.fromJson(Map<String, dynamic> json) =>
      _$CouponVOFromJson(json);

  final int id;
  final String name;
  final CouponType type;

  /// 面额（分）
  final int faceValue;

  /// 使用门槛（分）
  final int? threshold;
  final CouponValidType validType;
  final String? validStart;
  final String? validEnd;
  final int? validDays;
  final int totalQty;
  final int issuedQty;
  final int usedQty;
  final int perUserLimit;

  /// 适用商品：商品 ID 或商品类型（NULL / 空表示全部适用）
  final List<dynamic>? applicableScope;
  final int status;
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
    required this.type,
    required this.faceValue,
    required this.status,
    required this.receiveTime,
    this.threshold,
    this.expireTime,
    this.usedTime,
    this.usedOrderId,
    this.applicableScope,
  });

  factory UserCouponVO.fromJson(Map<String, dynamic> json) =>
      _$UserCouponVOFromJson(json);

  final int id;
  final int couponId;
  final String couponName;
  final CouponType type;

  /// 面额（分）
  final int faceValue;
  final int? threshold;

  @JsonKey(fromJson: _userCouponStatusFromJson, toJson: _userCouponStatusToJson)
  final UserCouponStatus status;
  final String receiveTime;
  final String? expireTime;
  final String? usedTime;
  final int? usedOrderId;

  /// 适用商品：商品 ID 或商品类型（NULL / 空表示全部适用）
  final List<dynamic>? applicableScope;

  Map<String, dynamic> toJson() => _$UserCouponVOToJson(this);
}

/// 优惠券批量发放表单
@JsonSerializable()
class CouponBatchDistributeForm {
  const CouponBatchDistributeForm({
    required this.couponId,
    required this.targetScope,
    this.levelCodes,
    this.userIds,
  });

  factory CouponBatchDistributeForm.fromJson(Map<String, dynamic> json) =>
      _$CouponBatchDistributeFormFromJson(json);

  final int couponId;

  /// 发放范围（all 全部 / level 按等级 / users 指定用户）
  final String targetScope;
  final List<String>? levelCodes;
  final List<int>? userIds;

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
