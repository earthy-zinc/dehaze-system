// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'package_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

BenefitOverrides _$BenefitOverridesFromJson(Map<String, dynamic> json) =>
    $checkedCreate('BenefitOverrides', json, ($checkedConvert) {
      final val = BenefitOverrides(
        monthlyDehazeQuota: $checkedConvert(
          'monthlyDehazeQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyDerainQuota: $checkedConvert(
          'monthlyDerainQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyDesnowQuota: $checkedConvert(
          'monthlyDesnowQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyLowlightQuota: $checkedConvert(
          'monthlyLowlightQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlySuperResolutionQuota: $checkedConvert(
          'monthlySuperResolutionQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyDenoiseQuota: $checkedConvert(
          'monthlyDenoiseQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyInpaintQuota: $checkedConvert(
          'monthlyInpaintQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyEvaluateQuota: $checkedConvert(
          'monthlyEvaluateQuota',
          (v) => (v as num?)?.toInt(),
        ),
        aiCreditsDaily: $checkedConvert(
          'aiCreditsDaily',
          (v) => (v as num?)?.toInt(),
        ),
        aiCreditsMonthly: $checkedConvert(
          'aiCreditsMonthly',
          (v) => (v as num?)?.toInt(),
        ),
        historyRetention: $checkedConvert(
          'historyRetention',
          (v) => (v as num?)?.toInt(),
        ),
        batchLimit: $checkedConvert('batchLimit', (v) => (v as num?)?.toInt()),
        priority: $checkedConvert('priority', (v) => (v as num?)?.toInt()),
        advancedParams: $checkedConvert(
          'advancedParams',
          (v) => (v as num?)?.toInt(),
        ),
        hdExport: $checkedConvert('hdExport', (v) => (v as num?)?.toInt()),
        reportExport: $checkedConvert(
          'reportExport',
          (v) => (v as num?)?.toInt(),
        ),
        batchDownload: $checkedConvert(
          'batchDownload',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$BenefitOverridesToJson(BenefitOverrides instance) =>
    <String, dynamic>{
      'monthlyDehazeQuota': ?instance.monthlyDehazeQuota,
      'monthlyDerainQuota': ?instance.monthlyDerainQuota,
      'monthlyDesnowQuota': ?instance.monthlyDesnowQuota,
      'monthlyLowlightQuota': ?instance.monthlyLowlightQuota,
      'monthlySuperResolutionQuota': ?instance.monthlySuperResolutionQuota,
      'monthlyDenoiseQuota': ?instance.monthlyDenoiseQuota,
      'monthlyInpaintQuota': ?instance.monthlyInpaintQuota,
      'monthlyEvaluateQuota': ?instance.monthlyEvaluateQuota,
      'aiCreditsDaily': ?instance.aiCreditsDaily,
      'aiCreditsMonthly': ?instance.aiCreditsMonthly,
      'historyRetention': ?instance.historyRetention,
      'batchLimit': ?instance.batchLimit,
      'priority': ?instance.priority,
      'advancedParams': ?instance.advancedParams,
      'hdExport': ?instance.hdExport,
      'reportExport': ?instance.reportExport,
      'batchDownload': ?instance.batchDownload,
    };

PackageForm _$PackageFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackageForm', json, ($checkedConvert) {
  final val = PackageForm(
    id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
    name: $checkedConvert('name', (v) => v as String),
    packageType: $checkedConvert('packageType', (v) => v as String),
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    salePrice: $checkedConvert('salePrice', (v) => (v as num).toInt()),
    levelCode: $checkedConvert('levelCode', (v) => v as String?),
    period: $checkedConvert('period', (v) => v as String?),
    periodDays: $checkedConvert('periodDays', (v) => (v as num?)?.toInt()),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    description: $checkedConvert('description', (v) => v as String?),
    benefitOverrides: $checkedConvert(
      'benefitOverrides',
      (v) => v == null
          ? null
          : BenefitOverrides.fromJson(v as Map<String, dynamic>),
    ),
    sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
    status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
  );
  return val;
});

Map<String, dynamic> _$PackageFormToJson(PackageForm instance) =>
    <String, dynamic>{
      'id': ?instance.id,
      'name': instance.name,
      'packageType': instance.packageType,
      'levelCode': ?instance.levelCode,
      'period': ?instance.period,
      'periodDays': ?instance.periodDays,
      'creditAmount': ?instance.creditAmount,
      'originalPrice': instance.originalPrice,
      'salePrice': instance.salePrice,
      'description': ?instance.description,
      'benefitOverrides': ?instance.benefitOverrides?.toJson(),
      'sort': ?instance.sort,
      'status': ?instance.status,
    };

PackagePageVO _$PackagePageVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackagePageVO', json, ($checkedConvert) {
  final val = PackagePageVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    packageType: $checkedConvert('packageType', (v) => v as String),
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    salePrice: $checkedConvert('salePrice', (v) => (v as num).toInt()),
    dailyPrice: $checkedConvert('dailyPrice', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => (v as num).toInt()),
    createTime: $checkedConvert('createTime', (v) => v as String),
    levelCode: $checkedConvert('levelCode', (v) => v as String?),
    levelName: $checkedConvert('levelName', (v) => v as String?),
    period: $checkedConvert('period', (v) => v as String?),
    periodDays: $checkedConvert('periodDays', (v) => (v as num?)?.toInt()),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    creditUnitPrice: $checkedConvert(
      'creditUnitPrice',
      (v) => (v as num?)?.toInt(),
    ),
    salesCount: $checkedConvert('salesCount', (v) => (v as num?)?.toInt()),
  );
  return val;
});

Map<String, dynamic> _$PackagePageVOToJson(PackagePageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'packageType': instance.packageType,
      'levelCode': ?instance.levelCode,
      'levelName': ?instance.levelName,
      'period': ?instance.period,
      'periodDays': ?instance.periodDays,
      'creditAmount': ?instance.creditAmount,
      'originalPrice': instance.originalPrice,
      'salePrice': instance.salePrice,
      'dailyPrice': instance.dailyPrice,
      'creditUnitPrice': ?instance.creditUnitPrice,
      'salesCount': ?instance.salesCount,
      'status': instance.status,
      'createTime': instance.createTime,
    };

PackageDetailVO _$PackageDetailVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PackageDetailVO', json, ($checkedConvert) {
  final val = PackageDetailVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    name: $checkedConvert('name', (v) => v as String),
    packageType: $checkedConvert('packageType', (v) => v as String),
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    salePrice: $checkedConvert('salePrice', (v) => (v as num).toInt()),
    dailyPrice: $checkedConvert('dailyPrice', (v) => (v as num).toInt()),
    benefits: $checkedConvert(
      'benefits',
      (v) => Map<String, int>.from(v as Map),
    ),
    salesCount: $checkedConvert('salesCount', (v) => (v as num).toInt()),
    levelCode: $checkedConvert('levelCode', (v) => v as String?),
    levelName: $checkedConvert('levelName', (v) => v as String?),
    period: $checkedConvert('period', (v) => v as String?),
    periodDays: $checkedConvert('periodDays', (v) => (v as num?)?.toInt()),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    creditUnitPrice: $checkedConvert(
      'creditUnitPrice',
      (v) => (v as num?)?.toInt(),
    ),
    description: $checkedConvert('description', (v) => v as String?),
    activePromotions: $checkedConvert(
      'activePromotions',
      (v) => (v as List<dynamic>?)
          ?.map((e) => PromotionVO.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$PackageDetailVOToJson(PackageDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'packageType': instance.packageType,
      'levelCode': ?instance.levelCode,
      'levelName': ?instance.levelName,
      'period': ?instance.period,
      'periodDays': ?instance.periodDays,
      'originalPrice': instance.originalPrice,
      'salePrice': instance.salePrice,
      'dailyPrice': instance.dailyPrice,
      'creditAmount': ?instance.creditAmount,
      'creditUnitPrice': ?instance.creditUnitPrice,
      'description': ?instance.description,
      'benefits': instance.benefits,
      'activePromotions': ?instance.activePromotions
          ?.map((e) => e.toJson())
          .toList(),
      'salesCount': instance.salesCount,
    };

PackageQuery _$PackageQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PackageQuery', json, ($checkedConvert) {
      final val = PackageQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt()),
        name: $checkedConvert('name', (v) => v as String?),
        packageType: $checkedConvert('packageType', (v) => v as String?),
        levelCode: $checkedConvert('levelCode', (v) => v as String?),
        period: $checkedConvert('period', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$PackageQueryToJson(PackageQuery instance) =>
    <String, dynamic>{
      'pageNum': ?instance.pageNum,
      'pageSize': ?instance.pageSize,
      'name': ?instance.name,
      'packageType': ?instance.packageType,
      'levelCode': ?instance.levelCode,
      'period': ?instance.period,
      'status': ?instance.status,
    };

PriceResult _$PriceResultFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('PriceResult', json, ($checkedConvert) {
  final val = PriceResult(
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    discountAmount: $checkedConvert(
      'discountAmount',
      (v) => (v as num).toInt(),
    ),
    couponAmount: $checkedConvert('couponAmount', (v) => (v as num).toInt()),
    payableAmount: $checkedConvert('payableAmount', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$PriceResultToJson(PriceResult instance) =>
    <String, dynamic>{
      'originalPrice': instance.originalPrice,
      'discountAmount': instance.discountAmount,
      'couponAmount': instance.couponAmount,
      'payableAmount': instance.payableAmount,
    };

PromotionVO _$PromotionVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'PromotionVO',
  json,
  ($checkedConvert) {
    final val = PromotionVO(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      name: $checkedConvert('name', (v) => v as String),
      type: $checkedConvert('type', (v) => v as String),
      startTime: $checkedConvert('startTime', (v) => v as String),
      endTime: $checkedConvert('endTime', (v) => v as String),
      description: $checkedConvert('description', (v) => v as String?),
      activityRules: $checkedConvert(
        'activityRules',
        (v) => v as Map<String, dynamic>?,
      ),
      newUserOnly: $checkedConvert('newUserOnly', (v) => (v as num?)?.toInt()),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$PromotionVOToJson(PromotionVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'type': instance.type,
      'description': ?instance.description,
      'startTime': instance.startTime,
      'endTime': instance.endTime,
      'activityRules': ?instance.activityRules,
      'newUserOnly': ?instance.newUserOnly,
      'status': ?instance.status,
    };

CouponForm _$CouponFormFromJson(Map<String, dynamic> json) => $checkedCreate(
  'CouponForm',
  json,
  ($checkedConvert) {
    final val = CouponForm(
      id: $checkedConvert('id', (v) => (v as num?)?.toInt()),
      name: $checkedConvert('name', (v) => v as String),
      type: $checkedConvert('type', (v) => $enumDecode(_$CouponTypeEnumMap, v)),
      faceValue: $checkedConvert('faceValue', (v) => (v as num).toInt()),
      validType: $checkedConvert(
        'validType',
        (v) => $enumDecode(_$CouponValidTypeEnumMap, v),
      ),
      totalQty: $checkedConvert('totalQty', (v) => (v as num).toInt()),
      perUserLimit: $checkedConvert('perUserLimit', (v) => (v as num).toInt()),
      threshold: $checkedConvert('threshold', (v) => (v as num?)?.toInt()),
      validStart: $checkedConvert('validStart', (v) => v as String?),
      validEnd: $checkedConvert('validEnd', (v) => v as String?),
      validDays: $checkedConvert('validDays', (v) => (v as num?)?.toInt()),
      applicableScope: $checkedConvert(
        'applicableScope',
        (v) => v as List<dynamic>?,
      ),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$CouponFormToJson(CouponForm instance) =>
    <String, dynamic>{
      'id': ?instance.id,
      'name': instance.name,
      'type': _$CouponTypeEnumMap[instance.type]!,
      'faceValue': instance.faceValue,
      'threshold': ?instance.threshold,
      'validType': _$CouponValidTypeEnumMap[instance.validType]!,
      'validStart': ?instance.validStart,
      'validEnd': ?instance.validEnd,
      'validDays': ?instance.validDays,
      'totalQty': instance.totalQty,
      'perUserLimit': instance.perUserLimit,
      'applicableScope': ?instance.applicableScope,
      'status': ?instance.status,
    };

const _$CouponTypeEnumMap = {
  CouponType.fullReduction: 'full_reduction',
  CouponType.discount: 'discount',
  CouponType.noThreshold: 'no_threshold',
  CouponType.trial: 'trial',
};

const _$CouponValidTypeEnumMap = {
  CouponValidType.fixed: 'fixed',
  CouponValidType.relative: 'relative',
};

CouponQuery _$CouponQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'CouponQuery',
  json,
  ($checkedConvert) {
    final val = CouponQuery(
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      name: $checkedConvert('name', (v) => v as String?),
      type: $checkedConvert('type', (v) => v as String?),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$CouponQueryToJson(CouponQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'name': ?instance.name,
      'type': ?instance.type,
      'status': ?instance.status,
    };

CouponVO _$CouponVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'CouponVO',
  json,
  ($checkedConvert) {
    final val = CouponVO(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      name: $checkedConvert('name', (v) => v as String),
      type: $checkedConvert('type', (v) => $enumDecode(_$CouponTypeEnumMap, v)),
      faceValue: $checkedConvert('faceValue', (v) => (v as num).toInt()),
      validType: $checkedConvert(
        'validType',
        (v) => $enumDecode(_$CouponValidTypeEnumMap, v),
      ),
      totalQty: $checkedConvert('totalQty', (v) => (v as num).toInt()),
      issuedQty: $checkedConvert('issuedQty', (v) => (v as num).toInt()),
      usedQty: $checkedConvert('usedQty', (v) => (v as num).toInt()),
      perUserLimit: $checkedConvert('perUserLimit', (v) => (v as num).toInt()),
      status: $checkedConvert('status', (v) => (v as num).toInt()),
      createTime: $checkedConvert('createTime', (v) => v as String),
      threshold: $checkedConvert('threshold', (v) => (v as num?)?.toInt()),
      validStart: $checkedConvert('validStart', (v) => v as String?),
      validEnd: $checkedConvert('validEnd', (v) => v as String?),
      validDays: $checkedConvert('validDays', (v) => (v as num?)?.toInt()),
      applicableScope: $checkedConvert(
        'applicableScope',
        (v) => v as List<dynamic>?,
      ),
    );
    return val;
  },
);

Map<String, dynamic> _$CouponVOToJson(CouponVO instance) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'type': _$CouponTypeEnumMap[instance.type]!,
  'faceValue': instance.faceValue,
  'threshold': ?instance.threshold,
  'validType': _$CouponValidTypeEnumMap[instance.validType]!,
  'validStart': ?instance.validStart,
  'validEnd': ?instance.validEnd,
  'validDays': ?instance.validDays,
  'totalQty': instance.totalQty,
  'issuedQty': instance.issuedQty,
  'usedQty': instance.usedQty,
  'perUserLimit': instance.perUserLimit,
  'applicableScope': ?instance.applicableScope,
  'status': instance.status,
  'createTime': instance.createTime,
};

UserCouponVO _$UserCouponVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('UserCouponVO', json, ($checkedConvert) {
  final val = UserCouponVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    couponId: $checkedConvert('couponId', (v) => (v as num).toInt()),
    couponName: $checkedConvert('couponName', (v) => v as String),
    type: $checkedConvert('type', (v) => $enumDecode(_$CouponTypeEnumMap, v)),
    faceValue: $checkedConvert('faceValue', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => _userCouponStatusFromJson(v)),
    receiveTime: $checkedConvert('receiveTime', (v) => v as String),
    threshold: $checkedConvert('threshold', (v) => (v as num?)?.toInt()),
    expireTime: $checkedConvert('expireTime', (v) => v as String?),
    usedTime: $checkedConvert('usedTime', (v) => v as String?),
    usedOrderId: $checkedConvert('usedOrderId', (v) => (v as num?)?.toInt()),
    applicableScope: $checkedConvert(
      'applicableScope',
      (v) => v as List<dynamic>?,
    ),
  );
  return val;
});

Map<String, dynamic> _$UserCouponVOToJson(UserCouponVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'couponId': instance.couponId,
      'couponName': instance.couponName,
      'type': _$CouponTypeEnumMap[instance.type]!,
      'faceValue': instance.faceValue,
      'threshold': ?instance.threshold,
      'status': _userCouponStatusToJson(instance.status),
      'receiveTime': instance.receiveTime,
      'expireTime': ?instance.expireTime,
      'usedTime': ?instance.usedTime,
      'usedOrderId': ?instance.usedOrderId,
      'applicableScope': ?instance.applicableScope,
    };

CouponBatchDistributeForm _$CouponBatchDistributeFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('CouponBatchDistributeForm', json, ($checkedConvert) {
  final val = CouponBatchDistributeForm(
    couponId: $checkedConvert('couponId', (v) => (v as num).toInt()),
    targetScope: $checkedConvert('targetScope', (v) => v as String),
    levelCodes: $checkedConvert(
      'levelCodes',
      (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
    ),
    userIds: $checkedConvert(
      'userIds',
      (v) => (v as List<dynamic>?)?.map((e) => (e as num).toInt()).toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$CouponBatchDistributeFormToJson(
  CouponBatchDistributeForm instance,
) => <String, dynamic>{
  'couponId': instance.couponId,
  'targetScope': instance.targetScope,
  'levelCodes': ?instance.levelCodes,
  'userIds': ?instance.userIds,
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
