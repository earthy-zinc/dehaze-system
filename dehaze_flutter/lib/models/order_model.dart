import 'package:json_annotation/json_annotation.dart';

part 'order_model.g.dart';

// ==================== 枚举 ====================

/// 订单状态
enum OrderStatus {
  @JsonValue(0)
  pending,
  @JsonValue(1)
  paid,
  @JsonValue(2)
  cancelled,
  @JsonValue(3)
  refunded,
  @JsonValue(4)
  expired,
  @JsonValue(5)
  failed,
}

extension OrderStatusExtension on OrderStatus {
  String get displayName {
    switch (this) {
      case OrderStatus.pending:
        return '待支付';
      case OrderStatus.paid:
        return '已支付';
      case OrderStatus.cancelled:
        return '已取消';
      case OrderStatus.refunded:
        return '已退款';
      case OrderStatus.expired:
        return '已过期';
      case OrderStatus.failed:
        return '支付失败';
    }
  }
}

/// 支付方式
enum PayMethod {
  @JsonValue('alipay')
  alipay,
  @JsonValue('wechat')
  wechat,
  @JsonValue('balance')
  balance,
}

extension PayMethodExtension on PayMethod {
  String get displayName {
    switch (this) {
      case PayMethod.alipay:
        return '支付宝';
      case PayMethod.wechat:
        return '微信支付';
      case PayMethod.balance:
        return '余额支付';
    }
  }
}

/// 退款状态
enum RefundStatus {
  @JsonValue(0)
  pending,
  @JsonValue(1)
  approved,
  @JsonValue(2)
  rejected,
  @JsonValue(3)
  refunded,
}

extension RefundStatusExtension on RefundStatus {
  String get displayName {
    switch (this) {
      case RefundStatus.pending:
        return '待审核';
      case RefundStatus.approved:
        return '已通过';
      case RefundStatus.rejected:
        return '已驳回';
      case RefundStatus.refunded:
        return '已退款';
    }
  }
}

// ==================== 请求表单 ====================

/// 订单创建表单
@JsonSerializable()
class OrderCreateForm {
  const OrderCreateForm({
    required this.packageId,
    this.couponId,
    this.autoRenew = false,
  });

  factory OrderCreateForm.fromJson(Map<String, dynamic> json) =>
      _$OrderCreateFormFromJson(json);

  final int packageId;
  final int? couponId;

  @JsonKey(name: 'autoRenew')
  final bool autoRenew;

  Map<String, dynamic> toJson() => _$OrderCreateFormToJson(this);
}

/// 我的订单查询参数
@JsonSerializable()
class MyOrderQuery {
  const MyOrderQuery({
    required this.pageNum,
    required this.pageSize,
    this.status,
  });

  factory MyOrderQuery.fromJson(Map<String, dynamic> json) =>
      _$MyOrderQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? status;

  Map<String, dynamic> toJson() => _$MyOrderQueryToJson(this);
}

/// 管理端订单查询参数
@JsonSerializable()
class OrderQuery {
  const OrderQuery({
    required this.pageNum,
    required this.pageSize,
    this.status,
    this.keyword,
    this.startDate,
    this.endDate,
  });

  factory OrderQuery.fromJson(Map<String, dynamic> json) =>
      _$OrderQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? status;
  final String? keyword;
  final String? startDate;
  final String? endDate;

  Map<String, dynamic> toJson() => _$OrderQueryToJson(this);
}

/// 支付请求
@JsonSerializable()
class PayRequest {
  const PayRequest({
    required this.orderId,
    required this.method,
    this.couponId,
  });

  factory PayRequest.fromJson(Map<String, dynamic> json) =>
      _$PayRequestFromJson(json);

  final int orderId;

  @JsonKey(name: 'payMethod')
  final String method;

  final int? couponId;

  Map<String, dynamic> toJson() => _$PayRequestToJson(this);
}

/// 退款申请表单
@JsonSerializable()
class RefundApplyForm {
  const RefundApplyForm({
    required this.orderId,
    required this.reason,
    required this.amount,
    this.description,
  });

  factory RefundApplyForm.fromJson(Map<String, dynamic> json) =>
      _$RefundApplyFormFromJson(json);

  final int orderId;
  final String reason;
  final double amount;
  final String? description;

  Map<String, dynamic> toJson() => _$RefundApplyFormToJson(this);
}

/// 退款审核表单
@JsonSerializable()
class RefundAuditForm {
  const RefundAuditForm({
    required this.refundId,
    required this.approved,
    this.remark,
  });

  factory RefundAuditForm.fromJson(Map<String, dynamic> json) =>
      _$RefundAuditFormFromJson(json);

  final int refundId;
  final bool approved;
  final String? remark;

  Map<String, dynamic> toJson() => _$RefundAuditFormToJson(this);
}

/// 退款查询参数
@JsonSerializable()
class RefundQuery {
  const RefundQuery({
    required this.pageNum,
    required this.pageSize,
    this.status,
    this.orderId,
    this.keyword,
  });

  factory RefundQuery.fromJson(Map<String, dynamic> json) =>
      _$RefundQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? status;
  final int? orderId;
  final String? keyword;

  Map<String, dynamic> toJson() => _$RefundQueryToJson(this);
}

/// 自动续费配置表单
@JsonSerializable()
class AutoRenewConfigForm {
  const AutoRenewConfigForm({
    required this.orderId,
    required this.enabled,
    this.couponId,
  });

  factory AutoRenewConfigForm.fromJson(Map<String, dynamic> json) =>
      _$AutoRenewConfigFormFromJson(json);

  final int orderId;
  final bool enabled;
  final int? couponId;

  Map<String, dynamic> toJson() => _$AutoRenewConfigFormToJson(this);
}

// ==================== VO ====================

/// 我的订单列表项
@JsonSerializable()
class MyOrderVO {
  const MyOrderVO({
    required this.id,
    required this.orderNo,
    required this.packageName,
    required this.packageLevel,
    required this.period,
    required this.amount,
    required this.status,
    required this.statusName,
    required this.createTime,
    this.payTime,
    this.expireTime,
    this.autoRenew = false,
  });

  factory MyOrderVO.fromJson(Map<String, dynamic> json) =>
      _$MyOrderVOFromJson(json);

  final int id;
  final String orderNo;
  final String packageName;
  final String packageLevel;
  final int period;
  final double amount;
  final int status;
  final String statusName;
  final String createTime;
  final String? payTime;
  final String? expireTime;

  @JsonKey(name: 'autoRenew')
  final bool autoRenew;

  Map<String, dynamic> toJson() => _$MyOrderVOToJson(this);
}

/// 管理端订单列表项
@JsonSerializable()
class OrderPageVO {
  const OrderPageVO({
    required this.id,
    required this.orderNo,
    required this.userId,
    required this.username,
    required this.nickname,
    required this.packageName,
    required this.packageLevel,
    required this.period,
    required this.amount,
    required this.status,
    required this.statusName,
    required this.createTime,
    this.payTime,
    this.expireTime,
    this.autoRenew = false,
    this.payMethod,
  });

  factory OrderPageVO.fromJson(Map<String, dynamic> json) =>
      _$OrderPageVOFromJson(json);

  final int id;
  final String orderNo;
  final int userId;
  final String username;
  final String nickname;
  final String packageName;
  final String packageLevel;
  final int period;
  final double amount;
  final int status;
  final String statusName;
  final String createTime;
  final String? payTime;
  final String? expireTime;

  @JsonKey(name: 'autoRenew')
  final bool autoRenew;

  final String? payMethod;

  Map<String, dynamic> toJson() => _$OrderPageVOToJson(this);
}

/// 支付记录 VO
@JsonSerializable()
class PaymentRecordVO {
  const PaymentRecordVO({
    required this.id,
    required this.orderId,
    required this.orderNo,
    required this.amount,
    required this.method,
    required this.methodName,
    required this.status,
    required this.statusName,
    required this.createTime,
    this.payTime,
    this.tradeNo,
  });

  factory PaymentRecordVO.fromJson(Map<String, dynamic> json) =>
      _$PaymentRecordVOFromJson(json);

  final int id;
  final int orderId;
  final String orderNo;
  final double amount;
  final String method;
  final String methodName;
  final int status;
  final String statusName;
  final String createTime;
  final String? payTime;
  final String? tradeNo;

  Map<String, dynamic> toJson() => _$PaymentRecordVOToJson(this);
}

/// 退款记录 VO
@JsonSerializable()
class RefundRecordVO {
  const RefundRecordVO({
    required this.id,
    required this.orderId,
    required this.orderNo,
    required this.userId,
    required this.username,
    required this.amount,
    required this.reason,
    required this.description,
    required this.status,
    required this.statusName,
    required this.createTime,
    this.auditTime,
    this.auditBy,
    this.auditRemark,
  });

  factory RefundRecordVO.fromJson(Map<String, dynamic> json) =>
      _$RefundRecordVOFromJson(json);

  final int id;
  final int orderId;
  final String orderNo;
  final int userId;
  final String username;
  final double amount;
  final String reason;
  final String? description;
  final int status;
  final String statusName;
  final String createTime;
  final String? auditTime;
  final String? auditBy;
  final String? auditRemark;

  Map<String, dynamic> toJson() => _$RefundRecordVOToJson(this);
}

/// 订单详情 VO
@JsonSerializable()
class OrderDetailVO {
  const OrderDetailVO({
    required this.id,
    required this.orderNo,
    required this.userId,
    required this.username,
    required this.nickname,
    required this.packageName,
    required this.packageLevel,
    required this.period,
    required this.originalPrice,
    required this.discount,
    required this.amount,
    required this.status,
    required this.statusName,
    required this.createTime,
    this.payTime,
    this.expireTime,
    this.autoRenew = false,
    this.payMethod,
    this.couponId,
    this.couponName,
    this.refundRecords,
    this.paymentRecords,
  });

  factory OrderDetailVO.fromJson(Map<String, dynamic> json) =>
      _$OrderDetailVOFromJson(json);

  final int id;
  final String orderNo;
  final int userId;
  final String username;
  final String nickname;
  final String packageName;
  final String packageLevel;
  final int period;
  final double originalPrice;
  final double discount;
  final double amount;
  final int status;
  final String statusName;
  final String createTime;
  final String? payTime;
  final String? expireTime;

  @JsonKey(name: 'autoRenew')
  final bool autoRenew;

  final String? payMethod;
  final int? couponId;
  final String? couponName;

  @JsonKey(name: 'refundRecords')
  final List<RefundRecordVO>? refundRecords;

  @JsonKey(name: 'paymentRecords')
  final List<PaymentRecordVO>? paymentRecords;

  Map<String, dynamic> toJson() => _$OrderDetailVOToJson(this);
}

/// 支付结果
@JsonSerializable()
class PayResult {
  const PayResult({
    required this.orderId,
    this.payUrl,
    this.qrCode,
    required this.status,
    this.paymentId,
  });

  factory PayResult.fromJson(Map<String, dynamic> json) =>
      _$PayResultFromJson(json);

  final int orderId;
  final String? payUrl;
  final String? qrCode;
  final String status;
  final int? paymentId;

  Map<String, dynamic> toJson() => _$PayResultToJson(this);
}

/// 月统计
@JsonSerializable()
class MonthlyStat {
  const MonthlyStat({
    required this.month,
    required this.orders,
    required this.amount,
  });

  factory MonthlyStat.fromJson(Map<String, dynamic> json) =>
      _$MonthlyStatFromJson(json);

  final String month;
  final int orders;
  final double amount;

  Map<String, dynamic> toJson() => _$MonthlyStatToJson(this);
}

/// 订单统计 VO
@JsonSerializable()
class OrderStatsVO {
  const OrderStatsVO({
    required this.totalOrders,
    required this.totalAmount,
    required this.paidOrders,
    required this.paidAmount,
    required this.pendingOrders,
    required this.pendingAmount,
    required this.refundedOrders,
    required this.refundedAmount,
    this.monthlyStats = const [],
  });

  factory OrderStatsVO.fromJson(Map<String, dynamic> json) =>
      _$OrderStatsVOFromJson(json);

  final int totalOrders;
  final double totalAmount;
  final int paidOrders;
  final double paidAmount;
  final int pendingOrders;
  final double pendingAmount;
  final int refundedOrders;
  final double refundedAmount;

  @JsonKey(name: 'monthlyStats')
  final List<MonthlyStat> monthlyStats;

  Map<String, dynamic> toJson() => _$OrderStatsVOToJson(this);
}

/// 自动续费配置 VO
@JsonSerializable()
class AutoRenewConfigVO {
  const AutoRenewConfigVO({
    required this.orderId,
    required this.enabled,
    this.couponId,
    this.couponName,
    this.nextRenewDate,
    this.packageId,
    this.packageName,
  });

  factory AutoRenewConfigVO.fromJson(Map<String, dynamic> json) =>
      _$AutoRenewConfigVOFromJson(json);

  final int orderId;
  final bool enabled;
  final int? couponId;
  final String? couponName;
  final String? nextRenewDate;
  final int? packageId;
  final String? packageName;

  Map<String, dynamic> toJson() => _$AutoRenewConfigVOToJson(this);
}
