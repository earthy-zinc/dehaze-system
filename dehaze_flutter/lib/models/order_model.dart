import 'package:json_annotation/json_annotation.dart';

part 'order_model.g.dart';

// ==================== 枚举 ====================

/// 订单状态
///
/// 后端以字符串枚举返回（pending/paid/completed/cancelled/refunding/refunded），
/// 未知值回退为待支付，避免后端新增状态导致解析失败。
enum OrderStatus {
  pending('pending', '待支付'),
  paid('paid', '已支付'),
  completed('completed', '已完成'),
  cancelled('cancelled', '已取消'),
  refunding('refunding', '退款中'),
  refunded('refunded', '已退款');

  const OrderStatus(this.value, this.displayName);

  final String value;
  final String displayName;

  static OrderStatus fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => OrderStatus.pending,
  );
}

/// 支付方式
enum PayMethod {
  @JsonValue('wechat')
  wechat,
  @JsonValue('alipay')
  alipay,
  @JsonValue('balance')
  balance,
  @JsonValue('combined')
  combined,
}

extension PayMethodExtension on PayMethod {
  String get displayName {
    switch (this) {
      case PayMethod.wechat:
        return '微信支付';
      case PayMethod.alipay:
        return '支付宝';
      case PayMethod.balance:
        return '余额支付';
      case PayMethod.combined:
        return '组合支付';
    }
  }
}

/// 退款状态（refunding 退款中 / refunded 退款成功 / refund_failed 退款失败）
enum RefundStatus {
  refunding('refunding', '退款中'),
  refunded('refunded', '退款成功'),
  refundFailed('refund_failed', '退款失败');

  const RefundStatus(this.value, this.displayName);

  final String value;
  final String displayName;

  static RefundStatus fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => RefundStatus.refunding,
  );
}

/// 售后原因类型
enum RefundReasonType {
  afterSale('after_sale', '售后问题'),
  forceMajeure('force_majeure', '不可抗原因'),
  merchant('merchant', '商家原因'),
  other('other', '其他');

  const RefundReasonType(this.value, this.displayName);

  final String value;
  final String displayName;

  static RefundReasonType fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => RefundReasonType.other,
  );
}

// ==================== JSON 转换辅助函数 ====================

OrderStatus _orderStatusFromJson(dynamic value) =>
    OrderStatus.fromValue(value is String ? value : null);

String _orderStatusToJson(OrderStatus status) => status.value;

RefundStatus _refundStatusFromJson(dynamic value) =>
    RefundStatus.fromValue(value is String ? value : null);

String _refundStatusToJson(RefundStatus status) => status.value;

RefundReasonType _refundReasonTypeFromJson(dynamic value) =>
    RefundReasonType.fromValue(value is String ? value : null);

String _refundReasonTypeToJson(RefundReasonType type) => type.value;

bool _autoRenewFromJson(dynamic value) => value == 1 || value == true;

int _autoRenewToJson(bool value) => value ? 1 : 0;

// ==================== 请求表单 ====================

/// 订单创建表单
@JsonSerializable()
class OrderCreateForm {
  const OrderCreateForm({
    required this.packageId,
    required this.payMethod,
    this.couponId,
    this.balanceAmount,
  });

  factory OrderCreateForm.fromJson(Map<String, dynamic> json) =>
      _$OrderCreateFormFromJson(json);

  final int packageId;
  final int? couponId;
  final PayMethod payMethod;

  /// 组合支付时余额部分金额（分）
  final int? balanceAmount;

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
    this.orderNo,
    this.keywords,
    this.status,
    this.packageType,
    this.payMethod,
    this.amountMin,
    this.amountMax,
    this.paidTimeStart,
    this.paidTimeEnd,
  });

  factory OrderQuery.fromJson(Map<String, dynamic> json) =>
      _$OrderQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? orderNo;
  final String? keywords;
  final String? status;

  /// 套餐类型（vip/credit）
  final String? packageType;
  final String? payMethod;

  /// 金额区间（分）
  final int? amountMin;
  final int? amountMax;
  final String? paidTimeStart;
  final String? paidTimeEnd;

  Map<String, dynamic> toJson() => _$OrderQueryToJson(this);
}

/// 支付请求
@JsonSerializable()
class PayRequest {
  const PayRequest({required this.payMethod, this.channel, this.balanceAmount});

  factory PayRequest.fromJson(Map<String, dynamic> json) =>
      _$PayRequestFromJson(json);

  final PayMethod payMethod;

  /// 组合支付时指定的第三方渠道（wechat/alipay），组合支付必填
  final String? channel;

  /// 组合支付时余额部分金额（分）
  final int? balanceAmount;

  Map<String, dynamic> toJson() => _$PayRequestToJson(this);
}

/// 退款申请表单
@JsonSerializable()
class RefundApplyForm {
  const RefundApplyForm({required this.reasonType, this.customReason});

  factory RefundApplyForm.fromJson(Map<String, dynamic> json) =>
      _$RefundApplyFormFromJson(json);

  @JsonKey(fromJson: _refundReasonTypeFromJson, toJson: _refundReasonTypeToJson)
  final RefundReasonType reasonType;

  /// 补充说明，非空时后端以 reasonType + ":" + customReason 拼接写入
  final String? customReason;

  Map<String, dynamic> toJson() => _$RefundApplyFormToJson(this);
}

/// 退款审核表单
@JsonSerializable()
class RefundAuditForm {
  const RefundAuditForm({required this.approved, required this.remark});

  factory RefundAuditForm.fromJson(Map<String, dynamic> json) =>
      _$RefundAuditFormFromJson(json);

  final bool approved;
  final String remark;

  Map<String, dynamic> toJson() => _$RefundAuditFormToJson(this);
}

/// 退款查询参数
@JsonSerializable()
class RefundQuery {
  const RefundQuery({
    required this.pageNum,
    required this.pageSize,
    this.orderNo,
    this.keywords,
    this.status,
    this.applyTimeStart,
    this.applyTimeEnd,
  });

  factory RefundQuery.fromJson(Map<String, dynamic> json) =>
      _$RefundQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? orderNo;
  final String? keywords;

  /// 退款状态（refunding/refunded/refund_failed）
  final String? status;
  final String? applyTimeStart;
  final String? applyTimeEnd;

  Map<String, dynamic> toJson() => _$RefundQueryToJson(this);
}

/// 自动续费配置表单
@JsonSerializable()
class AutoRenewConfigForm {
  const AutoRenewConfigForm({
    required this.packageId,
    required this.payMethod,
    required this.enabled,
  });

  factory AutoRenewConfigForm.fromJson(Map<String, dynamic> json) =>
      _$AutoRenewConfigFormFromJson(json);

  final int packageId;
  final PayMethod payMethod;
  final bool enabled;

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
    this.packageLevel,
    required this.packageType,
    required this.payableAmount,
    required this.paidAmount,
    required this.status,
    required this.createTime,
    this.creditAmount,
    this.payMethod,
    this.paidTime,
    this.packageExpireTime,
  });

  factory MyOrderVO.fromJson(Map<String, dynamic> json) =>
      _$MyOrderVOFromJson(json);

  final int id;
  final String orderNo;
  final String packageName;

  /// 商品对应会员等级（积分卡订单为 NULL）
  final String? packageLevel;

  /// 套餐类型（vip/credit）
  final String packageType;

  /// 积分卡额度（仅积分卡订单有值）
  final int? creditAmount;

  /// 应付金额（分）
  final int payableAmount;

  /// 实付金额（分）
  final int paidAmount;

  final String? payMethod;

  @JsonKey(fromJson: _orderStatusFromJson, toJson: _orderStatusToJson)
  final OrderStatus status;

  final String createTime;
  final String? paidTime;
  final String? packageExpireTime;

  Map<String, dynamic> toJson() => _$MyOrderVOToJson(this);
}

/// 管理端订单列表项 = 我的订单字段 + 用户与价格明细
@JsonSerializable()
class OrderPageVO {
  const OrderPageVO({
    required this.id,
    required this.orderNo,
    required this.packageName,
    this.packageLevel,
    required this.packageType,
    required this.payableAmount,
    required this.paidAmount,
    required this.status,
    required this.createTime,
    required this.userId,
    required this.username,
    required this.originalPrice,
    required this.discountAmount,
    required this.couponAmount,
    this.creditAmount,
    this.payMethod,
    this.paidTime,
    this.packageExpireTime,
  });

  factory OrderPageVO.fromJson(Map<String, dynamic> json) =>
      _$OrderPageVOFromJson(json);

  final int id;
  final String orderNo;
  final String packageName;

  /// 商品对应会员等级（积分卡订单为 NULL）
  final String? packageLevel;
  final String packageType;
  final int? creditAmount;

  /// 应付金额（分）
  final int payableAmount;

  /// 实付金额（分）
  final int paidAmount;

  final String? payMethod;

  @JsonKey(fromJson: _orderStatusFromJson, toJson: _orderStatusToJson)
  final OrderStatus status;

  final String createTime;
  final String? paidTime;
  final String? packageExpireTime;

  final int userId;
  final String username;

  /// 原价（分）
  final int originalPrice;

  /// 促销折扣（分）
  final int discountAmount;

  /// 优惠券抵扣（分）
  final int couponAmount;

  Map<String, dynamic> toJson() => _$OrderPageVOToJson(this);
}

/// 支付流水 VO
@JsonSerializable()
class PaymentRecordVO {
  const PaymentRecordVO({
    required this.id,
    required this.paymentNo,
    required this.amount,
    required this.status,
    required this.createTime,
    this.channel,
    this.callbackTime,
  });

  factory PaymentRecordVO.fromJson(Map<String, dynamic> json) =>
      _$PaymentRecordVOFromJson(json);

  final int id;

  /// 支付渠道流水号
  final String paymentNo;

  /// 支付渠道（wechat/alipay/balance/combined）
  final String? channel;

  /// 支付金额（分）
  final int amount;

  /// 支付状态（1 处理中 / 2 成功 / 3 失败）
  final int status;

  /// 回调时间
  final String? callbackTime;
  final String createTime;

  Map<String, dynamic> toJson() => _$PaymentRecordVOToJson(this);
}

/// 退款记录 VO
@JsonSerializable()
class RefundRecordVO {
  const RefundRecordVO({
    required this.id,
    required this.refundNo,
    required this.orderId,
    required this.orderNo,
    required this.userId,
    required this.username,
    required this.refundAmount,
    required this.reasonType,
    required this.reason,
    required this.status,
    required this.applyTime,
    this.usedDays,
    this.usedCredits,
    this.channel,
    this.channelRefundNo,
    this.auditTime,
    this.auditorId,
    this.auditRemark,
    this.refundTime,
    this.errorMessage,
  });

  factory RefundRecordVO.fromJson(Map<String, dynamic> json) =>
      _$RefundRecordVOFromJson(json);

  final int id;

  /// 退款单号
  final String refundNo;
  final int orderId;
  final String orderNo;
  final int userId;
  final String username;

  /// 退款金额（分）
  final int refundAmount;

  @JsonKey(fromJson: _refundReasonTypeFromJson, toJson: _refundReasonTypeToJson)
  final RefundReasonType reasonType;

  final String reason;

  /// 会员卡：申请时已使用天数
  final int? usedDays;

  /// 积分卡：申请时已消耗积分
  final int? usedCredits;

  @JsonKey(fromJson: _refundStatusFromJson, toJson: _refundStatusToJson)
  final RefundStatus status;

  /// 退款渠道
  final String? channel;

  /// 渠道退款流水号
  final String? channelRefundNo;

  /// 申请时间
  final String applyTime;
  final String? auditTime;

  /// 审核人 ID
  final int? auditorId;
  final String? auditRemark;
  final String? refundTime;
  final String? errorMessage;

  Map<String, dynamic> toJson() => _$RefundRecordVOToJson(this);
}

/// 订单详情 VO
///
/// 对齐后端 OrderDetailVO = OrderPageVO + 详情扩展字段。
@JsonSerializable()
class OrderDetailVO {
  const OrderDetailVO({
    required this.id,
    required this.orderNo,
    required this.packageName,
    this.packageLevel,
    required this.packageType,
    required this.payableAmount,
    required this.paidAmount,
    required this.status,
    required this.createTime,
    required this.originalPrice,
    this.creditAmount,
    this.payMethod,
    this.paidTime,
    this.packageExpireTime,
    this.userId,
    this.username,
    this.discountAmount,
    this.couponAmount,
    this.expireTime,
    this.effectiveTime,
    this.cancelReason,
    this.autoRenew = false,
    this.paymentRecords,
    this.refundRecord,
  });

  factory OrderDetailVO.fromJson(Map<String, dynamic> json) =>
      _$OrderDetailVOFromJson(json);

  final int id;
  final String orderNo;
  final String packageName;

  /// 商品对应会员等级（积分卡订单为 NULL）
  final String? packageLevel;
  final String packageType;
  final int? creditAmount;

  /// 应付金额（分）
  final int payableAmount;

  /// 实付金额（分）
  final int paidAmount;

  final String? payMethod;

  @JsonKey(fromJson: _orderStatusFromJson, toJson: _orderStatusToJson)
  final OrderStatus status;

  final String createTime;
  final String? paidTime;
  final String? packageExpireTime;

  final int? userId;
  final String? username;

  /// 原价（分）
  final int originalPrice;

  /// 促销折扣（分）
  final int? discountAmount;

  /// 优惠券抵扣（分）
  final int? couponAmount;

  /// 订单超时时间
  final String? expireTime;

  /// 权益生效时间
  final String? effectiveTime;

  final String? cancelReason;

  @JsonKey(
    name: 'isAutoRenew',
    fromJson: _autoRenewFromJson,
    toJson: _autoRenewToJson,
  )
  final bool autoRenew;

  final List<PaymentRecordVO>? paymentRecords;
  final RefundRecordVO? refundRecord;

  Map<String, dynamic> toJson() => _$OrderDetailVOToJson(this);
}

/// 支付结果
@JsonSerializable()
class PayResult {
  const PayResult({
    required this.orderNo,
    required this.payMethod,
    required this.paid,
    this.payUrl,
    this.qrCode,
  });

  factory PayResult.fromJson(Map<String, dynamic> json) =>
      _$PayResultFromJson(json);

  final String orderNo;
  final PayMethod payMethod;

  /// 是否已支付完成（余额支付同步完成时为 true）
  final bool paid;

  /// 第三方支付链接
  final String? payUrl;

  /// 二维码内容
  final String? qrCode;

  Map<String, dynamic> toJson() => _$PayResultToJson(this);
}

/// 订单统计 VO
@JsonSerializable()
class OrderStatsVO {
  const OrderStatsVO({
    required this.totalOrders,
    required this.totalRevenue,
    required this.totalRefund,
    required this.refundRate,
    this.statusDistribution,
    this.payMethodDistribution,
    this.packageDistribution,
    this.dailyStats,
  });

  factory OrderStatsVO.fromJson(Map<String, dynamic> json) =>
      _$OrderStatsVOFromJson(json);

  final int totalOrders;

  /// 累计收入（分）
  final int totalRevenue;

  /// 累计退款（分）
  final int totalRefund;

  /// 退款率
  final double refundRate;

  /// 各订单状态分布（状态 → 数量）
  final Map<String, int>? statusDistribution;

  /// 各支付方式分布（支付方式 → 数量）
  final Map<String, int>? payMethodDistribution;

  /// 各套餐分布
  final List<OrderPackageStatItem>? packageDistribution;

  /// 每日统计
  final List<OrderDailyStatItem>? dailyStats;

  Map<String, dynamic> toJson() => _$OrderStatsVOToJson(this);
}

/// 套餐维度统计项
@JsonSerializable()
class OrderPackageStatItem {
  const OrderPackageStatItem({
    required this.packageId,
    required this.packageName,
    required this.count,
    required this.revenue,
  });

  factory OrderPackageStatItem.fromJson(Map<String, dynamic> json) =>
      _$OrderPackageStatItemFromJson(json);

  final int packageId;
  final String packageName;
  final int count;

  /// 收入（分）
  final int revenue;

  Map<String, dynamic> toJson() => _$OrderPackageStatItemToJson(this);
}

/// 每日统计项
@JsonSerializable()
class OrderDailyStatItem {
  const OrderDailyStatItem({
    required this.date,
    required this.count,
    required this.revenue,
  });

  factory OrderDailyStatItem.fromJson(Map<String, dynamic> json) =>
      _$OrderDailyStatItemFromJson(json);

  final String date;
  final int count;

  /// 收入（分）
  final int revenue;

  Map<String, dynamic> toJson() => _$OrderDailyStatItemToJson(this);
}

/// 自动续费配置 VO
@JsonSerializable()
class AutoRenewConfigVO {
  const AutoRenewConfigVO({
    required this.userId,
    required this.packageId,
    required this.packageName,
    required this.payMethod,
    required this.enabled,
    this.nextRenewTime,
    this.failCount,
    this.closeReason,
  });

  factory AutoRenewConfigVO.fromJson(Map<String, dynamic> json) =>
      _$AutoRenewConfigVOFromJson(json);

  final int userId;
  final int packageId;
  final String packageName;
  final String payMethod;
  final bool enabled;

  /// 下次扣款时间
  final String? nextRenewTime;

  /// 连续失败次数
  final int? failCount;

  /// 关闭原因
  final String? closeReason;

  Map<String, dynamic> toJson() => _$AutoRenewConfigVOToJson(this);
}
