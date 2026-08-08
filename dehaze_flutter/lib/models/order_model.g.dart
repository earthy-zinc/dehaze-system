// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'order_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

OrderCreateForm _$OrderCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderCreateForm', json, ($checkedConvert) {
      final val = OrderCreateForm(
        packageId: $checkedConvert('packageId', (v) => (v as num).toInt()),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
        autoRenew: $checkedConvert('autoRenew', (v) => v as bool? ?? false),
      );
      return val;
    });

Map<String, dynamic> _$OrderCreateFormToJson(OrderCreateForm instance) =>
    <String, dynamic>{
      'packageId': instance.packageId,
      if (instance.couponId case final value?) 'couponId': value,
      'autoRenew': instance.autoRenew,
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
      if (instance.status case final value?) 'status': value,
    };

OrderQuery _$OrderQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderQuery', json, ($checkedConvert) {
      final val = OrderQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num).toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => v as String?),
        keyword: $checkedConvert('keyword', (v) => v as String?),
        startDate: $checkedConvert('startDate', (v) => v as String?),
        endDate: $checkedConvert('endDate', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$OrderQueryToJson(OrderQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.status case final value?) 'status': value,
      if (instance.keyword case final value?) 'keyword': value,
      if (instance.startDate case final value?) 'startDate': value,
      if (instance.endDate case final value?) 'endDate': value,
    };

PayRequest _$PayRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PayRequest', json, ($checkedConvert) {
      final val = PayRequest(
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        method: $checkedConvert('payMethod', (v) => v as String),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
      );
      return val;
    }, fieldKeyMap: const {'method': 'payMethod'});

Map<String, dynamic> _$PayRequestToJson(PayRequest instance) =>
    <String, dynamic>{
      'orderId': instance.orderId,
      'payMethod': instance.method,
      if (instance.couponId case final value?) 'couponId': value,
    };

RefundApplyForm _$RefundApplyFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundApplyForm', json, ($checkedConvert) {
      final val = RefundApplyForm(
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        reason: $checkedConvert('reason', (v) => v as String),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        description: $checkedConvert('description', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundApplyFormToJson(RefundApplyForm instance) =>
    <String, dynamic>{
      'orderId': instance.orderId,
      'reason': instance.reason,
      'amount': instance.amount,
      if (instance.description case final value?) 'description': value,
    };

RefundAuditForm _$RefundAuditFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundAuditForm', json, ($checkedConvert) {
      final val = RefundAuditForm(
        refundId: $checkedConvert('refundId', (v) => (v as num).toInt()),
        approved: $checkedConvert('approved', (v) => v as bool),
        remark: $checkedConvert('remark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundAuditFormToJson(RefundAuditForm instance) =>
    <String, dynamic>{
      'refundId': instance.refundId,
      'approved': instance.approved,
      if (instance.remark case final value?) 'remark': value,
    };

RefundQuery _$RefundQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundQuery', json, ($checkedConvert) {
      final val = RefundQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num).toInt()),
        pageSize: $checkedConvert('pageSize', (v) => (v as num).toInt()),
        status: $checkedConvert('status', (v) => v as String?),
        orderId: $checkedConvert('orderId', (v) => (v as num?)?.toInt()),
        keyword: $checkedConvert('keyword', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundQueryToJson(RefundQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.status case final value?) 'status': value,
      if (instance.orderId case final value?) 'orderId': value,
      if (instance.keyword case final value?) 'keyword': value,
    };

AutoRenewConfigForm _$AutoRenewConfigFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AutoRenewConfigForm', json, ($checkedConvert) {
      final val = AutoRenewConfigForm(
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        enabled: $checkedConvert('enabled', (v) => v as bool),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$AutoRenewConfigFormToJson(
  AutoRenewConfigForm instance,
) => <String, dynamic>{
  'orderId': instance.orderId,
  'enabled': instance.enabled,
  if (instance.couponId case final value?) 'couponId': value,
};

MyOrderVO _$MyOrderVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MyOrderVO', json, ($checkedConvert) {
      final val = MyOrderVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        packageName: $checkedConvert('packageName', (v) => v as String),
        packageLevel: $checkedConvert('packageLevel', (v) => v as String),
        period: $checkedConvert('period', (v) => (v as num).toInt()),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        payTime: $checkedConvert('payTime', (v) => v as String?),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        autoRenew: $checkedConvert('autoRenew', (v) => v as bool? ?? false),
      );
      return val;
    });

Map<String, dynamic> _$MyOrderVOToJson(MyOrderVO instance) => <String, dynamic>{
  'id': instance.id,
  'orderNo': instance.orderNo,
  'packageName': instance.packageName,
  'packageLevel': instance.packageLevel,
  'period': instance.period,
  'amount': instance.amount,
  'status': instance.status,
  'statusName': instance.statusName,
  'createTime': instance.createTime,
  if (instance.payTime case final value?) 'payTime': value,
  if (instance.expireTime case final value?) 'expireTime': value,
  'autoRenew': instance.autoRenew,
};

OrderPageVO _$OrderPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderPageVO', json, ($checkedConvert) {
      final val = OrderPageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
        packageName: $checkedConvert('packageName', (v) => v as String),
        packageLevel: $checkedConvert('packageLevel', (v) => v as String),
        period: $checkedConvert('period', (v) => (v as num).toInt()),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        payTime: $checkedConvert('payTime', (v) => v as String?),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        autoRenew: $checkedConvert('autoRenew', (v) => v as bool? ?? false),
        payMethod: $checkedConvert('payMethod', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$OrderPageVOToJson(OrderPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'orderNo': instance.orderNo,
      'userId': instance.userId,
      'username': instance.username,
      'nickname': instance.nickname,
      'packageName': instance.packageName,
      'packageLevel': instance.packageLevel,
      'period': instance.period,
      'amount': instance.amount,
      'status': instance.status,
      'statusName': instance.statusName,
      'createTime': instance.createTime,
      if (instance.payTime case final value?) 'payTime': value,
      if (instance.expireTime case final value?) 'expireTime': value,
      'autoRenew': instance.autoRenew,
      if (instance.payMethod case final value?) 'payMethod': value,
    };

PaymentRecordVO _$PaymentRecordVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PaymentRecordVO', json, ($checkedConvert) {
      final val = PaymentRecordVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        method: $checkedConvert('method', (v) => v as String),
        methodName: $checkedConvert('methodName', (v) => v as String),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        payTime: $checkedConvert('payTime', (v) => v as String?),
        tradeNo: $checkedConvert('tradeNo', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$PaymentRecordVOToJson(PaymentRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'orderId': instance.orderId,
      'orderNo': instance.orderNo,
      'amount': instance.amount,
      'method': instance.method,
      'methodName': instance.methodName,
      'status': instance.status,
      'statusName': instance.statusName,
      'createTime': instance.createTime,
      if (instance.payTime case final value?) 'payTime': value,
      if (instance.tradeNo case final value?) 'tradeNo': value,
    };

RefundRecordVO _$RefundRecordVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RefundRecordVO', json, ($checkedConvert) {
      final val = RefundRecordVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        reason: $checkedConvert('reason', (v) => v as String),
        description: $checkedConvert('description', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        auditTime: $checkedConvert('auditTime', (v) => v as String?),
        auditBy: $checkedConvert('auditBy', (v) => v as String?),
        auditRemark: $checkedConvert('auditRemark', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RefundRecordVOToJson(RefundRecordVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'orderId': instance.orderId,
      'orderNo': instance.orderNo,
      'userId': instance.userId,
      'username': instance.username,
      'amount': instance.amount,
      'reason': instance.reason,
      if (instance.description case final value?) 'description': value,
      'status': instance.status,
      'statusName': instance.statusName,
      'createTime': instance.createTime,
      if (instance.auditTime case final value?) 'auditTime': value,
      if (instance.auditBy case final value?) 'auditBy': value,
      if (instance.auditRemark case final value?) 'auditRemark': value,
    };

OrderDetailVO _$OrderDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('OrderDetailVO', json, ($checkedConvert) {
      final val = OrderDetailVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        orderNo: $checkedConvert('orderNo', (v) => v as String),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
        packageName: $checkedConvert('packageName', (v) => v as String),
        packageLevel: $checkedConvert('packageLevel', (v) => v as String),
        period: $checkedConvert('period', (v) => (v as num).toInt()),
        originalPrice: $checkedConvert(
          'originalPrice',
          (v) => (v as num).toDouble(),
        ),
        discount: $checkedConvert('discount', (v) => (v as num).toDouble()),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        statusName: $checkedConvert('statusName', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        payTime: $checkedConvert('payTime', (v) => v as String?),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        autoRenew: $checkedConvert('autoRenew', (v) => v as bool? ?? false),
        payMethod: $checkedConvert('payMethod', (v) => v as String?),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
        couponName: $checkedConvert('couponName', (v) => v as String?),
        refundRecords: $checkedConvert(
          'refundRecords',
          (v) => (v as List<dynamic>?)
              ?.map((e) => RefundRecordVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
        paymentRecords: $checkedConvert(
          'paymentRecords',
          (v) => (v as List<dynamic>?)
              ?.map((e) => PaymentRecordVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$OrderDetailVOToJson(
  OrderDetailVO instance,
) => <String, dynamic>{
  'id': instance.id,
  'orderNo': instance.orderNo,
  'userId': instance.userId,
  'username': instance.username,
  'nickname': instance.nickname,
  'packageName': instance.packageName,
  'packageLevel': instance.packageLevel,
  'period': instance.period,
  'originalPrice': instance.originalPrice,
  'discount': instance.discount,
  'amount': instance.amount,
  'status': instance.status,
  'statusName': instance.statusName,
  'createTime': instance.createTime,
  if (instance.payTime case final value?) 'payTime': value,
  if (instance.expireTime case final value?) 'expireTime': value,
  'autoRenew': instance.autoRenew,
  if (instance.payMethod case final value?) 'payMethod': value,
  if (instance.couponId case final value?) 'couponId': value,
  if (instance.couponName case final value?) 'couponName': value,
  if (instance.refundRecords?.map((e) => e.toJson()).toList() case final value?)
    'refundRecords': value,
  if (instance.paymentRecords?.map((e) => e.toJson()).toList()
      case final value?)
    'paymentRecords': value,
};

PayResult _$PayResultFromJson(Map<String, dynamic> json) =>
    $checkedCreate('PayResult', json, ($checkedConvert) {
      final val = PayResult(
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        payUrl: $checkedConvert('payUrl', (v) => v as String?),
        qrCode: $checkedConvert('qrCode', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String),
        paymentId: $checkedConvert('paymentId', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$PayResultToJson(PayResult instance) => <String, dynamic>{
  'orderId': instance.orderId,
  if (instance.payUrl case final value?) 'payUrl': value,
  if (instance.qrCode case final value?) 'qrCode': value,
  'status': instance.status,
  if (instance.paymentId case final value?) 'paymentId': value,
};

MonthlyStat _$MonthlyStatFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MonthlyStat', json, ($checkedConvert) {
      final val = MonthlyStat(
        month: $checkedConvert('month', (v) => v as String),
        orders: $checkedConvert('orders', (v) => (v as num).toInt()),
        amount: $checkedConvert('amount', (v) => (v as num).toDouble()),
      );
      return val;
    });

Map<String, dynamic> _$MonthlyStatToJson(MonthlyStat instance) =>
    <String, dynamic>{
      'month': instance.month,
      'orders': instance.orders,
      'amount': instance.amount,
    };

OrderStatsVO _$OrderStatsVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('OrderStatsVO', json, ($checkedConvert) {
  final val = OrderStatsVO(
    totalOrders: $checkedConvert('totalOrders', (v) => (v as num).toInt()),
    totalAmount: $checkedConvert('totalAmount', (v) => (v as num).toDouble()),
    paidOrders: $checkedConvert('paidOrders', (v) => (v as num).toInt()),
    paidAmount: $checkedConvert('paidAmount', (v) => (v as num).toDouble()),
    pendingOrders: $checkedConvert('pendingOrders', (v) => (v as num).toInt()),
    pendingAmount: $checkedConvert(
      'pendingAmount',
      (v) => (v as num).toDouble(),
    ),
    refundedOrders: $checkedConvert(
      'refundedOrders',
      (v) => (v as num).toInt(),
    ),
    refundedAmount: $checkedConvert(
      'refundedAmount',
      (v) => (v as num).toDouble(),
    ),
    monthlyStats: $checkedConvert(
      'monthlyStats',
      (v) =>
          (v as List<dynamic>?)
              ?.map((e) => MonthlyStat.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
    ),
  );
  return val;
});

Map<String, dynamic> _$OrderStatsVOToJson(OrderStatsVO instance) =>
    <String, dynamic>{
      'totalOrders': instance.totalOrders,
      'totalAmount': instance.totalAmount,
      'paidOrders': instance.paidOrders,
      'paidAmount': instance.paidAmount,
      'pendingOrders': instance.pendingOrders,
      'pendingAmount': instance.pendingAmount,
      'refundedOrders': instance.refundedOrders,
      'refundedAmount': instance.refundedAmount,
      'monthlyStats': instance.monthlyStats.map((e) => e.toJson()).toList(),
    };

AutoRenewConfigVO _$AutoRenewConfigVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AutoRenewConfigVO', json, ($checkedConvert) {
      final val = AutoRenewConfigVO(
        orderId: $checkedConvert('orderId', (v) => (v as num).toInt()),
        enabled: $checkedConvert('enabled', (v) => v as bool),
        couponId: $checkedConvert('couponId', (v) => (v as num?)?.toInt()),
        couponName: $checkedConvert('couponName', (v) => v as String?),
        nextRenewDate: $checkedConvert('nextRenewDate', (v) => v as String?),
        packageId: $checkedConvert('packageId', (v) => (v as num?)?.toInt()),
        packageName: $checkedConvert('packageName', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AutoRenewConfigVOToJson(AutoRenewConfigVO instance) =>
    <String, dynamic>{
      'orderId': instance.orderId,
      'enabled': instance.enabled,
      if (instance.couponId case final value?) 'couponId': value,
      if (instance.couponName case final value?) 'couponName': value,
      if (instance.nextRenewDate case final value?) 'nextRenewDate': value,
      if (instance.packageId case final value?) 'packageId': value,
      if (instance.packageName case final value?) 'packageName': value,
    };
