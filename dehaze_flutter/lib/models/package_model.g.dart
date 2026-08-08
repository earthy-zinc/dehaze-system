// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'package_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

BenefitVO _$BenefitVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('BenefitVO', json, ($checkedConvert) {
  final val = BenefitVO(
    levelCode: $checkedConvert('levelCode', (v) => v as String),
    levelName: $checkedConvert('levelName', (v) => v as String),
    growthMin: $checkedConvert('growthMin', (v) => (v as num).toInt()),
    growthMax: $checkedConvert('growthMax', (v) => (v as num).toInt()),
    monthlyDehazeQuota: $checkedConvert(
      'monthlyDehazeQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyEvaluateQuota: $checkedConvert(
      'monthlyEvaluateQuota',
      (v) => (v as num).toInt(),
    ),
    historyRetention: $checkedConvert(
      'historyRetention',
      (v) => (v as num).toInt(),
    ),
    batchLimit: $checkedConvert('batchLimit', (v) => (v as num).toInt()),
    priority: $checkedConvert('priority', (v) => (v as num).toInt()),
    advancedParams: $checkedConvert(
      'advancedParams',
      (v) => (v as num).toInt(),
    ),
    hdExport: $checkedConvert('hdExport', (v) => (v as num).toInt()),
    reportExport: $checkedConvert('reportExport', (v) => (v as num).toInt()),
    batchDownload: $checkedConvert('batchDownload', (v) => (v as num).toInt()),
    sort: $checkedConvert('sort', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$BenefitVOToJson(BenefitVO instance) => <String, dynamic>{
  'levelCode': instance.levelCode,
  'levelName': instance.levelName,
  'growthMin': instance.growthMin,
  'growthMax': instance.growthMax,
  'monthlyDehazeQuota': instance.monthlyDehazeQuota,
  'monthlyEvaluateQuota': instance.monthlyEvaluateQuota,
  'historyRetention': instance.historyRetention,
  'batchLimit': instance.batchLimit,
  'priority': instance.priority,
  'advancedParams': instance.advancedParams,
  'hdExport': instance.hdExport,
  'reportExport': instance.reportExport,
  'batchDownload': instance.batchDownload,
  'sort': instance.sort,
  'status': instance.status,
};

PackageForm _$PackageFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackageForm', json, ($checkedConvert) {
  final val = PackageForm(
    id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
    name: $checkedConvert('name', (v) => v as String),
    level: $checkedConvert('level', (v) => v as String),
    period: $checkedConvert('period', (v) => v as String),
    originalPrice: $checkedConvert(
      'originalPrice',
      (v) => (v as num).toDouble(),
    ),
    currentPrice: $checkedConvert('currentPrice', (v) => (v as num).toDouble()),
    description: $checkedConvert('description', (v) => v as String?),
    features: $checkedConvert(
      'features',
      (v) =>
          (v as List<dynamic>?)?.map((e) => e as String).toList() ?? const [],
    ),
    status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
    benefits: $checkedConvert('benefits', (v) => v as Map<String, dynamic>?),
  );
  return val;
});

Map<String, dynamic> _$PackageFormToJson(PackageForm instance) =>
    <String, dynamic>{
      if (instance.id case final value?) 'id': value,
      'name': instance.name,
      'level': instance.level,
      'period': instance.period,
      'originalPrice': instance.originalPrice,
      'currentPrice': instance.currentPrice,
      if (instance.description case final value?) 'description': value,
      'features': instance.features,
      if (instance.status case final value?) 'status': value,
      if (instance.sort case final value?) 'sort': value,
      if (instance.benefits case final value?) 'benefits': value,
    };

PackagePageVO _$PackagePageVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackagePageVO', json, ($checkedConvert) {
  final val = PackagePageVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    level: $checkedConvert('level', (v) => v as String),
    levelName: $checkedConvert('levelName', (v) => v as String),
    period: $checkedConvert('period', (v) => v as String),
    periodName: $checkedConvert('periodName', (v) => v as String),
    originalPrice: $checkedConvert(
      'originalPrice',
      (v) => (v as num).toDouble(),
    ),
    currentPrice: $checkedConvert('currentPrice', (v) => (v as num).toDouble()),
    description: $checkedConvert('description', (v) => v as String?),
    features: $checkedConvert(
      'features',
      (v) =>
          (v as List<dynamic>?)?.map((e) => e as String).toList() ?? const [],
    ),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
    sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$PackagePageVOToJson(PackagePageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'level': instance.level,
      'levelName': instance.levelName,
      'period': instance.period,
      'periodName': instance.periodName,
      'originalPrice': instance.originalPrice,
      'currentPrice': instance.currentPrice,
      if (instance.description case final value?) 'description': value,
      'features': instance.features,
      'status': instance.status,
      if (instance.sort case final value?) 'sort': value,
      'createTime': instance.createTime,
    };

PackageDetailVO _$PackageDetailVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackageDetailVO', json, ($checkedConvert) {
  final val = PackageDetailVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    level: $checkedConvert('level', (v) => v as String),
    levelName: $checkedConvert('levelName', (v) => v as String),
    period: $checkedConvert('period', (v) => v as String),
    periodName: $checkedConvert('periodName', (v) => v as String),
    originalPrice: $checkedConvert(
      'originalPrice',
      (v) => (v as num).toDouble(),
    ),
    currentPrice: $checkedConvert('currentPrice', (v) => (v as num).toDouble()),
    description: $checkedConvert('description', (v) => v as String?),
    features: $checkedConvert(
      'features',
      (v) =>
          (v as List<dynamic>?)?.map((e) => e as String).toList() ?? const [],
    ),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
    sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
    benefits: $checkedConvert(
      'benefits',
      (v) => (v as List<dynamic>?)
          ?.map((e) => BenefitVO.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$PackageDetailVOToJson(PackageDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'level': instance.level,
      'levelName': instance.levelName,
      'period': instance.period,
      'periodName': instance.periodName,
      'originalPrice': instance.originalPrice,
      'currentPrice': instance.currentPrice,
      if (instance.description case final value?) 'description': value,
      'features': instance.features,
      'status': instance.status,
      if (instance.sort case final value?) 'sort': value,
      'createTime': instance.createTime,
      if (instance.benefits?.map((e) => e.toJson()).toList() case final value?)
        'benefits': value,
    };

PackageQuery _$PackageQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PackageQuery', json, ($checkedConvert) {
      final val = PackageQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt()),
        level: $checkedConvert('level', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        keyword: $checkedConvert('keyword', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PackageQueryToJson(PackageQuery instance) =>
    <String, dynamic>{
      if (instance.pageNum case final value?) 'pageNum': value,
      if (instance.pageSize case final value?) 'pageSize': value,
      if (instance.level case final value?) 'level': value,
      if (instance.status case final value?) 'status': value,
      if (instance.keyword case final value?) 'keyword': value,
    };

PriceResult _$PriceResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PriceResult', json, ($checkedConvert) {
      final val = PriceResult(
        originalPrice: $checkedConvert(
          'originalPrice',
          (v) => (v as num).toDouble(),
        ),
        currentPrice: $checkedConvert(
          'currentPrice',
          (v) => (v as num).toDouble(),
        ),
        discount: $checkedConvert('discount', (v) => (v as num).toDouble()),
        couponDiscount: $checkedConvert(
          'couponDiscount',
          (v) => (v as num).toDouble(),
        ),
        finalPrice: $checkedConvert('finalPrice', (v) => (v as num).toDouble()),
        appliedCoupons: $checkedConvert(
          'appliedCoupons',
          (v) => (v as List<dynamic>?)
              ?.map((e) => UserCouponVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$PriceResultToJson(PriceResult instance) =>
    <String, dynamic>{
      'originalPrice': instance.originalPrice,
      'currentPrice': instance.currentPrice,
      'discount': instance.discount,
      'couponDiscount': instance.couponDiscount,
      'finalPrice': instance.finalPrice,
      if (instance.appliedCoupons?.map((e) => e.toJson()).toList()
          case final value?)
        'appliedCoupons': value,
    };

PromotionVO _$PromotionVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PromotionVO', json, ($checkedConvert) {
      final val = PromotionVO(
        title: $checkedConvert('title', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String?),
        startDate: $checkedConvert('startDate', (v) => v as String),
        endDate: $checkedConvert('endDate', (v) => v as String),
        discount: $checkedConvert('discount', (v) => (v as num).toDouble()),
        active: $checkedConvert('active', (v) => v as bool),
      );
      return val;
    });

Map<String, dynamic> _$PromotionVOToJson(PromotionVO instance) =>
    <String, dynamic>{
      'title': instance.title,
      if (instance.description case final value?) 'description': value,
      'startDate': instance.startDate,
      'endDate': instance.endDate,
      'discount': instance.discount,
      'active': instance.active,
    };

CouponForm _$CouponFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CouponForm', json, ($checkedConvert) {
      final val = CouponForm(
        code: $checkedConvert('code', (v) => v as String?),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        value: $checkedConvert('value', (v) => (v as num).toDouble()),
        validType: $checkedConvert('validType', (v) => v as String),
        validDays: $checkedConvert('validDays', (v) => (v as num?)?.toInt()),
        validStart: $checkedConvert('validStart', (v) => v as String?),
        validEnd: $checkedConvert('validEnd', (v) => v as String?),
        minAmount: $checkedConvert('minAmount', (v) => (v as num?)?.toDouble()),
        maxDiscount: $checkedConvert(
          'maxDiscount',
          (v) => (v as num?)?.toDouble(),
        ),
        total: $checkedConvert('total', (v) => (v as num).toInt()),
        perUser: $checkedConvert('perUser', (v) => (v as num).toInt()),
        description: $checkedConvert('description', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        scope: $checkedConvert('scope', (v) => (v as num?)?.toInt()),
        packageIds: $checkedConvert(
          'packageIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$CouponFormToJson(CouponForm instance) =>
    <String, dynamic>{
      if (instance.code case final value?) 'code': value,
      'name': instance.name,
      'type': instance.type,
      'value': instance.value,
      'validType': instance.validType,
      if (instance.validDays case final value?) 'validDays': value,
      if (instance.validStart case final value?) 'validStart': value,
      if (instance.validEnd case final value?) 'validEnd': value,
      if (instance.minAmount case final value?) 'minAmount': value,
      if (instance.maxDiscount case final value?) 'maxDiscount': value,
      'total': instance.total,
      'perUser': instance.perUser,
      if (instance.description case final value?) 'description': value,
      if (instance.status case final value?) 'status': value,
      if (instance.scope case final value?) 'scope': value,
      if (instance.packageIds case final value?) 'packageIds': value,
    };

CouponQuery _$CouponQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CouponQuery', json, ($checkedConvert) {
      final val = CouponQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt()),
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
        keyword: $checkedConvert('keyword', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$CouponQueryToJson(CouponQuery instance) =>
    <String, dynamic>{
      if (instance.pageNum case final value?) 'pageNum': value,
      if (instance.pageSize case final value?) 'pageSize': value,
      if (instance.type case final value?) 'type': value,
      if (instance.status case final value?) 'status': value,
      if (instance.keyword case final value?) 'keyword': value,
    };

CouponVO _$CouponVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CouponVO', json, ($checkedConvert) {
      final val = CouponVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        code: $checkedConvert('code', (v) => v as String),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String),
        typeName: $checkedConvert('typeName', (v) => v as String),
        value: $checkedConvert('value', (v) => (v as num).toDouble()),
        validType: $checkedConvert('validType', (v) => v as String),
        validDays: $checkedConvert('validDays', (v) => (v as num?)?.toInt()),
        validStart: $checkedConvert('validStart', (v) => v as String?),
        validEnd: $checkedConvert('validEnd', (v) => v as String?),
        minAmount: $checkedConvert('minAmount', (v) => (v as num?)?.toDouble()),
        maxDiscount: $checkedConvert(
          'maxDiscount',
          (v) => (v as num?)?.toDouble(),
        ),
        total: $checkedConvert('total', (v) => (v as num).toInt()),
        issued: $checkedConvert('issued', (v) => (v as num).toInt()),
        remaining: $checkedConvert('remaining', (v) => (v as num).toInt()),
        perUser: $checkedConvert('perUser', (v) => (v as num).toInt()),
        description: $checkedConvert('description', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        scope: $checkedConvert('scope', (v) => (v as num?)?.toInt()),
        packageIds: $checkedConvert(
          'packageIds',
          (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
        ),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$CouponVOToJson(CouponVO instance) => <String, dynamic>{
  'id': instance.id,
  'code': instance.code,
  'name': instance.name,
  'type': instance.type,
  'typeName': instance.typeName,
  'value': instance.value,
  'validType': instance.validType,
  if (instance.validDays case final value?) 'validDays': value,
  if (instance.validStart case final value?) 'validStart': value,
  if (instance.validEnd case final value?) 'validEnd': value,
  if (instance.minAmount case final value?) 'minAmount': value,
  if (instance.maxDiscount case final value?) 'maxDiscount': value,
  'total': instance.total,
  'issued': instance.issued,
  'remaining': instance.remaining,
  'perUser': instance.perUser,
  if (instance.description case final value?) 'description': value,
  'status': instance.status,
  if (instance.scope case final value?) 'scope': value,
  if (instance.packageIds case final value?) 'packageIds': value,
  'createTime': instance.createTime,
};

UserCouponVO _$UserCouponVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('UserCouponVO', json, ($checkedConvert) {
  final val = UserCouponVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    couponId: $checkedConvert('couponId', (v) => (v as num).toInt()),
    couponName: $checkedConvert('couponName', (v) => v as String),
    couponType: $checkedConvert('couponType', (v) => v as String),
    couponValue: $checkedConvert('couponValue', (v) => (v as num).toDouble()),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
    validStart: $checkedConvert('validStart', (v) => v as String),
    validEnd: $checkedConvert('validEnd', (v) => v as String),
    minAmount: $checkedConvert('minAmount', (v) => (v as num?)?.toDouble()),
    maxDiscount: $checkedConvert('maxDiscount', (v) => (v as num?)?.toDouble()),
    usedTime: $checkedConvert('usedTime', (v) => v as String?),
    orderId: $checkedConvert('orderId', (v) => (v as num?)?.toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$UserCouponVOToJson(UserCouponVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'couponId': instance.couponId,
      'couponName': instance.couponName,
      'couponType': instance.couponType,
      'couponValue': instance.couponValue,
      'status': instance.status,
      'validStart': instance.validStart,
      'validEnd': instance.validEnd,
      if (instance.minAmount case final value?) 'minAmount': value,
      if (instance.maxDiscount case final value?) 'maxDiscount': value,
      if (instance.usedTime case final value?) 'usedTime': value,
      if (instance.orderId case final value?) 'orderId': value,
      'createTime': instance.createTime,
    };

CouponBatchDistributeForm _$CouponBatchDistributeFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('CouponBatchDistributeForm', json, ($checkedConvert) {
  final val = CouponBatchDistributeForm(
    couponId: $checkedConvert('couponId', (v) => (v as num).toInt()),
    userIds: $checkedConvert(
      'userIds',
      (v) => (v as List<dynamic>).map((e) => (e as num).toInt()).toList(),
    ),
    count: $checkedConvert('count', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$CouponBatchDistributeFormToJson(
  CouponBatchDistributeForm instance,
) => <String, dynamic>{
  'couponId': instance.couponId,
  'userIds': instance.userIds,
  'count': instance.count,
};

MonthlyRevenue _$MonthlyRevenueFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MonthlyRevenue', json, ($checkedConvert) {
      final val = MonthlyRevenue(
        month: $checkedConvert('month', (v) => v as String),
        revenue: $checkedConvert('revenue', (v) => (v as num).toDouble()),
        orders: $checkedConvert('orders', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$MonthlyRevenueToJson(MonthlyRevenue instance) =>
    <String, dynamic>{
      'month': instance.month,
      'revenue': instance.revenue,
      'orders': instance.orders,
    };

PackageSales _$PackageSalesFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PackageSales', json, ($checkedConvert) {
      final val = PackageSales(
        packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
        packageName: $checkedConvert('packageName', (v) => v as String),
        sales: $checkedConvert('sales', (v) => (v as num).toInt()),
        revenue: $checkedConvert('revenue', (v) => (v as num).toDouble()),
      );
      return val;
    });

Map<String, dynamic> _$PackageSalesToJson(PackageSales instance) =>
    <String, dynamic>{
      'packageId': instance.packageId,
      'packageName': instance.packageName,
      'sales': instance.sales,
      'revenue': instance.revenue,
    };

SalesStatsVO _$SalesStatsVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('SalesStatsVO', json, ($checkedConvert) {
  final val = SalesStatsVO(
    totalSales: $checkedConvert('totalSales', (v) => (v as num).toDouble()),
    totalOrders: $checkedConvert('totalOrders', (v) => (v as num).toInt()),
    activeSubscriptions: $checkedConvert(
      'activeSubscriptions',
      (v) => (v as num).toInt(),
    ),
    renewalRate: $checkedConvert('renewalRate', (v) => (v as num).toDouble()),
    monthlyRevenue: $checkedConvert(
      'monthlyRevenue',
      (v) =>
          (v as List<dynamic>?)
              ?.map((e) => MonthlyRevenue.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
    ),
    topPackages: $checkedConvert(
      'topPackages',
      (v) =>
          (v as List<dynamic>?)
              ?.map((e) => PackageSales.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
    ),
  );
  return val;
});

Map<String, dynamic> _$SalesStatsVOToJson(SalesStatsVO instance) =>
    <String, dynamic>{
      'totalSales': instance.totalSales,
      'totalOrders': instance.totalOrders,
      'activeSubscriptions': instance.activeSubscriptions,
      'renewalRate': instance.renewalRate,
      'monthlyRevenue': instance.monthlyRevenue.map((e) => e.toJson()).toList(),
      'topPackages': instance.topPackages.map((e) => e.toJson()).toList(),
    };
