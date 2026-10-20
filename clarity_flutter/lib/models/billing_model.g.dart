// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'billing_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

BalanceVO _$BalanceVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'BalanceVO',
  json,
  ($checkedConvert) {
    final val = BalanceVO(
      userId: $checkedConvert('userId', (v) => (v as num).toInt()),
      creditsBalance: $checkedConvert('creditsBalance', (v) => v as String),
      arrearsStatus: $checkedConvert('arrearsStatus', (v) => v as bool),
      dailyUsed: $checkedConvert('dailyUsed', (v) => (v as num).toInt()),
      dailyLimit: $checkedConvert('dailyLimit', (v) => (v as num).toInt()),
      monthlyUsed: $checkedConvert('monthlyUsed', (v) => (v as num).toInt()),
      monthlyLimit: $checkedConvert('monthlyLimit', (v) => (v as num).toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$BalanceVOToJson(BalanceVO instance) => <String, dynamic>{
  'userId': instance.userId,
  'creditsBalance': instance.creditsBalance,
  'arrearsStatus': instance.arrearsStatus,
  'dailyUsed': instance.dailyUsed,
  'dailyLimit': instance.dailyLimit,
  'monthlyUsed': instance.monthlyUsed,
  'monthlyLimit': instance.monthlyLimit,
};
