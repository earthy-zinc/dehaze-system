import 'package:json_annotation/json_annotation.dart';

part 'billing_model.g.dart';

/// 用户余额与配额使用概况（对齐后端 BalanceResult / SDK BalanceVO）
@JsonSerializable()
class BalanceVO {
  const BalanceVO({
    required this.userId,
    required this.creditsBalance,
    required this.arrearsStatus,
    required this.dailyUsed,
    required this.dailyLimit,
    required this.monthlyUsed,
    required this.monthlyLimit,
  });

  factory BalanceVO.fromJson(Map<String, dynamic> json) =>
      _$BalanceVOFromJson(json);

  final int userId;

  /// 积分余额（后端 Decimal 按字符串下发，原样展示）
  final String creditsBalance;

  /// 是否欠费
  final bool arrearsStatus;

  /// 今日已用积分
  final int dailyUsed;

  /// 日积分限额
  final int dailyLimit;

  /// 本月已用积分
  final int monthlyUsed;

  /// 月积分限额
  final int monthlyLimit;

  Map<String, dynamic> toJson() => _$BalanceVOToJson(this);
}
