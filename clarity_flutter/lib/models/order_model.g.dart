// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'order_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

OrderCreateForm _$OrderCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderCreateForm', json, ($checkedConvert) {
      final val = OrderCreateForm(
        packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
        payMethod: $checkedConvert(
          'payMethod',
          (v) => $enumDecode(_$PayMethodEnumMap, v),
        ),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
        balanceAmount: $checkedConvert(
          'balanceAmount',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$OrderCreateFormToJson(OrderCreateForm instance) =>
    <String, dynamic>{
      'packageId': instance.packageId,
      'couponId': ?instance.couponId,
      'payMethod': _$PayMethodEnumMap[instance.payMethod]!,
      'balanceAmount': ?instance.balanceAmount,
    };

const _$PayMethodEnumMap = {
  PayMethod.wechat: 'wechat',
  PayMethod.alipay: 'alipay',
  PayMethod.balance: 'balance',
  PayMethod.combined: 'combined',
};

MyOrderQuery _$MyOrderQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MyOrderQuery', json, ($checkedConvert) {
      final val = MyOrderQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num).toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$MyOrderQueryToJson(MyOrderQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'status': ?instance.status,
    };

OrderQuery _$OrderQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderQuery', json, ($checkedConvert) {
      final val = OrderQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num).toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String?),
        keywords: $checkedConvert('keywords', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
        packageType: $checkedConvert('packageType', (v) => v as String?),
        payMethod: $checkedConvert('payMethod', (v) => v as String?),
        amountMin: $checkedConvert('amountMin', (v) => (v as num?)?.toInt()),
        amountMax: $checkedConvert('amountMax', (v) => (v as num?)?.toInt()),
        paidTimeStart: $checkedConvert('paidTimeStart', (v) => v as String?),
        paidTimeEnd: $checkedConvert('paidTimeEnd', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$OrderQueryToJson(OrderQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'orderNo': ?instance.orderNo,
      'keywords': ?instance.keywords,
      'status': ?instance.status,
      'packageType': ?instance.packageType,
      'payMethod': ?instance.payMethod,
      'amountMin': ?instance.amountMin,
      'amountMax': ?instance.amountMax,
      'paidTimeStart': ?instance.paidTimeStart,
      'paidTimeEnd': ?instance.paidTimeEnd,
    };

PayRequest _$PayRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PayRequest', json, ($checkedConvert) {
      final val = PayRequest(
        payMethod: $checkedConvert(
          'payMethod',
          (v) => $enumDecode(_$PayMethodEnumMap, v),
        ),
        channel: $checkedConvert('channel', (v) => v as String?),
        balanceAmount: $checkedConvert(
          'balanceAmount',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$PayRequestToJson(PayRequest instance) =>
    <String, dynamic>{
      'payMethod': _$PayMethodEnumMap[instance.payMethod]!,
      'channel': ?instance.channel,
      'balanceAmount': ?instance.balanceAmount,
    };

RefundApplyForm _$RefundApplyFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundApplyForm', json, ($checkedConvert) {
      final val = RefundApplyForm(
        reasonType: $checkedConvert(
          'reasonType',
          (v) => _refundReasonTypeFromJson(v),
        ),
        customReason: $checkedConvert('customReason', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundApplyFormToJson(RefundApplyForm instance) =>
    <String, dynamic>{
      'reasonType': _refundReasonTypeToJson(instance.reasonType),
      'customReason': ?instance.customReason,
    };

RefundAuditForm _$RefundAuditFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundAuditForm', json, ($checkedConvert) {
      final val = RefundAuditForm(
        approved: $checkedConvert('approved', (v) => v as bool),
        remark: $checkedConvert('remark', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$RefundAuditFormToJson(RefundAuditForm instance) =>
    <String, dynamic>{'approved': instance.approved, 'remark': instance.remark};

RefundQuery _$RefundQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundQuery', json, ($checkedConvert) {
      final val = RefundQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num).toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String?),
        keywords: $checkedConvert('keywords', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
        applyTimeStart: $checkedConvert('applyTimeStart', (v) => v as String?),
        applyTimeEnd: $checkedConvert('applyTimeEnd', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundQueryToJson(RefundQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'orderNo': ?instance.orderNo,
      'keywords': ?instance.keywords,
      'status': ?instance.status,
      'applyTimeStart': ?instance.applyTimeStart,
      'applyTimeEnd': ?instance.applyTimeEnd,
    };

AutoRenewConfigForm _$AutoRenewConfigFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AutoRenewConfigForm', json, ($checkedConvert) {
      final val = AutoRenewConfigForm(
        packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
        payMethod: $checkedConvert(
          'payMethod',
          (v) => $enumDecode(_$PayMethodEnumMap, v),
        ),
        enabled: $checkedConvert('enabled', (v) => v as bool),
      );
      return val;
    });

Map<String, dynamic> _$AutoRenewConfigFormToJson(
  AutoRenewConfigForm instance,
) => <String, dynamic>{
  'packageId': instance.packageId,
  'payMethod': _$PayMethodEnumMap[instance.payMethod]!,
  'enabled': instance.enabled,
};

MyOrderVO _$MyOrderVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('MyOrderVO', json, ($checkedConvert) {
  final val = MyOrderVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    orderNo: $checkedConvert('orderNo', (v) => v as String),
    packageName: $checkedConvert('packageName', (v) => v as String),
    packageLevel: $checkedConvert('packageLevel', (v) => v as String?),
    packageType: $checkedConvert('packageType', (v) => v as String),
    payableAmount: $checkedConvert('payableAmount', (v) => (v as num).toInt()),
    paidAmount: $checkedConvert('paidAmount', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => _orderStatusFromJson(v)),
    createTime: $checkedConvert('createTime', (v) => v as String),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    payMethod: $checkedConvert('payMethod', (v) => v as String?),
    paidTime: $checkedConvert('paidTime', (v) => v as String?),
    packageExpireTime: $checkedConvert(
      'packageExpireTime',
      (v) => v as String?,
    ),
  );
  return val;
});

Map<String, dynamic> _$MyOrderVOToJson(MyOrderVO instance) => <String, dynamic>{
  'id': instance.id,
  'orderNo': instance.orderNo,
  'packageName': instance.packageName,
  'packageLevel': ?instance.packageLevel,
  'packageType': instance.packageType,
  'creditAmount': ?instance.creditAmount,
  'payableAmount': instance.payableAmount,
  'paidAmount': instance.paidAmount,
  'payMethod': ?instance.payMethod,
  'status': _orderStatusToJson(instance.status),
  'createTime': instance.createTime,
  'paidTime': ?instance.paidTime,
  'packageExpireTime': ?instance.packageExpireTime,
};

OrderPageVO _$OrderPageVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('OrderPageVO', json, ($checkedConvert) {
  final val = OrderPageVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    orderNo: $checkedConvert('orderNo', (v) => v as String),
    packageName: $checkedConvert('packageName', (v) => v as String),
    packageLevel: $checkedConvert('packageLevel', (v) => v as String?),
    packageType: $checkedConvert('packageType', (v) => v as String),
    payableAmount: $checkedConvert('payableAmount', (v) => (v as num).toInt()),
    paidAmount: $checkedConvert('paidAmount', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => _orderStatusFromJson(v)),
    createTime: $checkedConvert('createTime', (v) => v as String),
    userId: $checkedConvert('userId', (v) => (v as num).toInt()),
    username: $checkedConvert('username', (v) => v as String),
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    discountAmount: $checkedConvert(
      'discountAmount',
      (v) => (v as num).toInt(),
    ),
    couponAmount: $checkedConvert('couponAmount', (v) => (v as num).toInt()),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    payMethod: $checkedConvert('payMethod', (v) => v as String?),
    paidTime: $checkedConvert('paidTime', (v) => v as String?),
    packageExpireTime: $checkedConvert(
      'packageExpireTime',
      (v) => v as String?,
    ),
  );
  return val;
});

Map<String, dynamic> _$OrderPageVOToJson(OrderPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'orderNo': instance.orderNo,
      'packageName': instance.packageName,
      'packageLevel': ?instance.packageLevel,
      'packageType': instance.packageType,
      'creditAmount': ?instance.creditAmount,
      'payableAmount': instance.payableAmount,
      'paidAmount': instance.paidAmount,
      'payMethod': ?instance.payMethod,
      'status': _orderStatusToJson(instance.status),
      'createTime': instance.createTime,
      'paidTime': ?instance.paidTime,
      'packageExpireTime': ?instance.packageExpireTime,
      'userId': instance.userId,
      'username': instance.username,
      'originalPrice': instance.originalPrice,
      'discountAmount': instance.discountAmount,
      'couponAmount': instance.couponAmount,
    };

PaymentRecordVO _$PaymentRecordVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PaymentRecordVO', json, ($checkedConvert) {
      final val = PaymentRecordVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        paymentNo: $checkedConvert('paymentNo', (v) => v as String),
        amount: $checkedConvert('amount', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String),
        channel: $checkedConvert('channel', (v) => v as String?),
        callbackTime: $checkedConvert('callbackTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PaymentRecordVOToJson(PaymentRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'paymentNo': instance.paymentNo,
      'channel': ?instance.channel,
      'amount': instance.amount,
      'status': instance.status,
      'callbackTime': ?instance.callbackTime,
      'createTime': instance.createTime,
    };

RefundRecordVO _$RefundRecordVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('RefundRecordVO', json, ($checkedConvert) {
  final val = RefundRecordVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    refundNo: $checkedConvert('refundNo', (v) => v as String),
    orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
    orderNo: $checkedConvert('orderNo', (v) => v as String),
    userId: $checkedConvert('userId', (v) => (v as num).toInt()),
    username: $checkedConvert('username', (v) => v as String),
    refundAmount: $checkedConvert('refundAmount', (v) => (v as num).toInt()),
    reasonType: $checkedConvert(
      'reasonType',
      (v) => _refundReasonTypeFromJson(v),
    ),
    reason: $checkedConvert('reason', (v) => v as String),
    status: $checkedConvert('status', (v) => _refundStatusFromJson(v)),
    applyTime: $checkedConvert('applyTime', (v) => v as String),
    usedDays: $checkedConvert('usedDays', (v) => (v as num?)?.toInt()),
    usedCredits: $checkedConvert('usedCredits', (v) => (v as num?)?.toInt()),
    channel: $checkedConvert('channel', (v) => v as String?),
    channelRefundNo: $checkedConvert('channelRefundNo', (v) => v as String?),
    auditTime: $checkedConvert('auditTime', (v) => v as String?),
    auditorId: $checkedConvert('auditorId', (v) => (v as num?)?.toInt()),
    auditRemark: $checkedConvert('auditRemark', (v) => v as String?),
    refundTime: $checkedConvert('refundTime', (v) => v as String?),
    errorMessage: $checkedConvert('errorMessage', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$RefundRecordVOToJson(RefundRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'refundNo': instance.refundNo,
      'orderId': instance.orderId,
      'orderNo': instance.orderNo,
      'userId': instance.userId,
      'username': instance.username,
      'refundAmount': instance.refundAmount,
      'reasonType': _refundReasonTypeToJson(instance.reasonType),
      'reason': instance.reason,
      'usedDays': ?instance.usedDays,
      'usedCredits': ?instance.usedCredits,
      'status': _refundStatusToJson(instance.status),
      'channel': ?instance.channel,
      'channelRefundNo': ?instance.channelRefundNo,
      'applyTime': instance.applyTime,
      'auditTime': ?instance.auditTime,
      'auditorId': ?instance.auditorId,
      'auditRemark': ?instance.auditRemark,
      'refundTime': ?instance.refundTime,
      'errorMessage': ?instance.errorMessage,
    };

OrderDetailVO _$OrderDetailVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('OrderDetailVO', json, ($checkedConvert) {
  final val = OrderDetailVO(
    id: $checkedConvert('id', (v) => (v as num).toInt()),
    orderNo: $checkedConvert('orderNo', (v) => v as String),
    packageName: $checkedConvert('packageName', (v) => v as String),
    packageLevel: $checkedConvert('packageLevel', (v) => v as String?),
    packageType: $checkedConvert('packageType', (v) => v as String),
    payableAmount: $checkedConvert('payableAmount', (v) => (v as num).toInt()),
    paidAmount: $checkedConvert('paidAmount', (v) => (v as num).toInt()),
    status: $checkedConvert('status', (v) => _orderStatusFromJson(v)),
    createTime: $checkedConvert('createTime', (v) => v as String),
    originalPrice: $checkedConvert('originalPrice', (v) => (v as num).toInt()),
    creditAmount: $checkedConvert('creditAmount', (v) => (v as num?)?.toInt()),
    payMethod: $checkedConvert('payMethod', (v) => v as String?),
    paidTime: $checkedConvert('paidTime', (v) => v as String?),
    packageExpireTime: $checkedConvert(
      'packageExpireTime',
      (v) => v as String?,
    ),
    userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
    username: $checkedConvert('username', (v) => v as String?),
    discountAmount: $checkedConvert(
      'discountAmount',
      (v) => (v as num?)?.toInt(),
    ),
    couponAmount: $checkedConvert('couponAmount', (v) => (v as num?)?.toInt()),
    expireTime: $checkedConvert('expireTime', (v) => v as String?),
    effectiveTime: $checkedConvert('effectiveTime', (v) => v as String?),
    cancelReason: $checkedConvert('cancelReason', (v) => v as String?),
    autoRenew: $checkedConvert(
      'isAutoRenew',
      (v) => v == null ? false : _autoRenewFromJson(v),
    ),
    paymentRecords: $checkedConvert(
      'paymentRecords',
      (v) => (v as List<dynamic>?)
          ?.map((e) => PaymentRecordVO.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
    refundRecord: $checkedConvert(
      'refundRecord',
      (v) =>
          v == null ? null : RefundRecordVO.fromJson(v as Map<String, dynamic>),
    ),
  );
  return val;
}, fieldKeyMap: const {'autoRenew': 'isAutoRenew'});

Map<String, dynamic> _$OrderDetailVOToJson(
  OrderDetailVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'orderNo': instance.orderNo,
  'packageName': instance.packageName,
  'packageLevel': ?instance.packageLevel,
  'packageType': instance.packageType,
  'creditAmount': ?instance.creditAmount,
  'payableAmount': instance.payableAmount,
  'paidAmount': instance.paidAmount,
  'payMethod': ?instance.payMethod,
  'status': _orderStatusToJson(instance.status),
  'createTime': instance.createTime,
  'paidTime': ?instance.paidTime,
  'packageExpireTime': ?instance.packageExpireTime,
  'userId': ?instance.userId,
  'username': ?instance.username,
  'originalPrice': instance.originalPrice,
  'discountAmount': ?instance.discountAmount,
  'couponAmount': ?instance.couponAmount,
  'expireTime': ?instance.expireTime,
  'effectiveTime': ?instance.effectiveTime,
  'cancelReason': ?instance.cancelReason,
  'isAutoRenew': _autoRenewToJson(instance.autoRenew),
  'paymentRecords': ?instance.paymentRecords?.map((e) => e.toJson()).toList(),
  'refundRecord': ?instance.refundRecord?.toJson(),
};

PayResult _$PayResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PayResult', json, ($checkedConvert) {
      final val = PayResult(
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        payMethod: $checkedConvert(
          'payMethod',
          (v) => $enumDecode(_$PayMethodEnumMap, v),
        ),
        paid: $checkedConvert('paid', (v) => v as bool),
        payUrl: $checkedConvert('payUrl', (v) => v as String?),
        qrCode: $checkedConvert('qrCode', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PayResultToJson(PayResult instance) => <String, dynamic>{
  'orderNo': instance.orderNo,
  'payMethod': _$PayMethodEnumMap[instance.payMethod]!,
  'paid': instance.paid,
  'payUrl': ?instance.payUrl,
  'qrCode': ?instance.qrCode,
};

OrderStatsVO _$OrderStatsVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('OrderStatsVO', json, ($checkedConvert) {
  final val = OrderStatsVO(
    totalOrders: $checkedConvert('totalOrders', (v) => (v as num).toInt()),
    totalRevenue: $checkedConvert('totalRevenue', (v) => (v as num).toInt()),
    totalRefund: $checkedConvert('totalRefund', (v) => (v as num).toInt()),
    refundRate: $checkedConvert('refundRate', (v) => (v as num).toDouble()),
    statusDistribution: $checkedConvert(
      'statusDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    payMethodDistribution: $checkedConvert(
      'payMethodDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    packageDistribution: $checkedConvert(
      'packageDistribution',
      (v) => (v as List<dynamic>?)
          ?.map((e) => OrderPackageStatItem.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
    dailyStats: $checkedConvert(
      'dailyStats',
      (v) => (v as List<dynamic>?)
          ?.map((e) => OrderDailyStatItem.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$OrderStatsVOToJson(OrderStatsVO instance) =>
    <String, dynamic>{
      'totalOrders': instance.totalOrders,
      'totalRevenue': instance.totalRevenue,
      'totalRefund': instance.totalRefund,
      'refundRate': instance.refundRate,
      'statusDistribution': ?instance.statusDistribution,
      'payMethodDistribution': ?instance.payMethodDistribution,
      'packageDistribution': ?instance.packageDistribution
          ?.map((e) => e.toJson())
          .toList(),
      'dailyStats': ?instance.dailyStats?.map((e) => e.toJson()).toList(),
    };

OrderPackageStatItem _$OrderPackageStatItemFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('OrderPackageStatItem', json, ($checkedConvert) {
  final val = OrderPackageStatItem(
    packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
    packageName: $checkedConvert('packageName', (v) => v as String),
    count: $checkedConvert('count', (v) => (v as num).toInt()),
    revenue: $checkedConvert('revenue', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$OrderPackageStatItemToJson(
  OrderPackageStatItem instance,
) => <String, dynamic>{
  'packageId': instance.packageId,
  'packageName': instance.packageName,
  'count': instance.count,
  'revenue': instance.revenue,
};

OrderDailyStatItem _$OrderDailyStatItemFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderDailyStatItem', json, ($checkedConvert) {
      final val = OrderDailyStatItem(
        date: $checkedConvert('date', (v) => v as String),
        count: $checkedConvert('count', (v) => (v as num).toInt()),
        revenue: $checkedConvert('revenue', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$OrderDailyStatItemToJson(OrderDailyStatItem instance) =>
    <String, dynamic>{
      'date': instance.date,
      'count': instance.count,
      'revenue': instance.revenue,
    };

AutoRenewConfigVO _$AutoRenewConfigVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AutoRenewConfigVO', json, ($checkedConvert) {
      final val = AutoRenewConfigVO(
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
        packageName: $checkedConvert('packageName', (v) => v as String),
        payMethod: $checkedConvert('payMethod', (v) => v as String),
        enabled: $checkedConvert('enabled', (v) => v as bool),
        nextRenewTime: $checkedConvert('nextRenewTime', (v) => v as String?),
        failCount: $checkedConvert('failCount', (v) => (v as num?)?.toInt()),
        closeReason: $checkedConvert('closeReason', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AutoRenewConfigVOToJson(AutoRenewConfigVO instance) =>
    <String, dynamic>{
      'userId': instance.userId,
      'packageId': instance.packageId,
      'packageName': instance.packageName,
      'payMethod': instance.payMethod,
      'enabled': instance.enabled,
      'nextRenewTime': ?instance.nextRenewTime,
      'failCount': ?instance.failCount,
      'closeReason': ?instance.closeReason,
    };
