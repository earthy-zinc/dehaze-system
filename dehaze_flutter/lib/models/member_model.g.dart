// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'member_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

MemberQuery _$MemberQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'MemberQuery',
  json,
  ($checkedConvert) {
    final val = MemberQuery(
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      keywords: $checkedConvert('keywords', (v) => v as String?),
      levelCode: $checkedConvert('levelCode', (v) => v as String?),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      expireTimeStart: $checkedConvert('expireTimeStart', (v) => v as String?),
      expireTimeEnd: $checkedConvert('expireTimeEnd', (v) => v as String?),
      growthMin: $checkedConvert('growthMin', (v) => (v as num?)?.toInt()),
      growthMax: $checkedConvert('growthMax', (v) => (v as num?)?.toInt()),
    );
    return val;
  },
);

Map<String, dynamic> _$MemberQueryToJson(MemberQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.keywords case final value?) 'keywords': value,
      if (instance.levelCode case final value?) 'levelCode': value,
      if (instance.status case final value?) 'status': value,
      if (instance.expireTimeStart case final value?) 'expireTimeStart': value,
      if (instance.expireTimeEnd case final value?) 'expireTimeEnd': value,
      if (instance.growthMin case final value?) 'growthMin': value,
      if (instance.growthMax case final value?) 'growthMax': value,
    };

GrowthLogQuery _$GrowthLogQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('GrowthLogQuery', json, ($checkedConvert) {
      final val = GrowthLogQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        changeType: $checkedConvert('changeType', (v) => v as String?),
        startTime: $checkedConvert('startTime', (v) => v as String?),
        endTime: $checkedConvert('endTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$GrowthLogQueryToJson(GrowthLogQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.changeType case final value?) 'changeType': value,
      if (instance.startTime case final value?) 'startTime': value,
      if (instance.endTime case final value?) 'endTime': value,
    };

MemberLevelAdjustForm _$MemberLevelAdjustFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('MemberLevelAdjustForm', json, ($checkedConvert) {
  final val = MemberLevelAdjustForm(
    levelCode: $checkedConvert('levelCode', (v) => v as String),
    expireTime: $checkedConvert('expireTime', (v) => v as String?),
    reason: $checkedConvert('reason', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$MemberLevelAdjustFormToJson(
  MemberLevelAdjustForm instance,
) => <String, dynamic>{
  'levelCode': instance.levelCode,
  if (instance.expireTime case final value?) 'expireTime': value,
  'reason': instance.reason,
};

MemberGrowthAdjustForm _$MemberGrowthAdjustFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('MemberGrowthAdjustForm', json, ($checkedConvert) {
  final val = MemberGrowthAdjustForm(
    changeValue: $checkedConvert('changeValue', (v) => (v as num).toInt()),
    reason: $checkedConvert('reason', (v) => v as String),
  );
  return val;
});

Map<String, dynamic> _$MemberGrowthAdjustFormToJson(
  MemberGrowthAdjustForm instance,
) => <String, dynamic>{
  'changeValue': instance.changeValue,
  'reason': instance.reason,
};

MemberStatusForm _$MemberStatusFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MemberStatusForm', json, ($checkedConvert) {
      final val = MemberStatusForm(
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        reason: $checkedConvert('reason', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$MemberStatusFormToJson(MemberStatusForm instance) =>
    <String, dynamic>{
      'status': instance.status,
      if (instance.reason case final value?) 'reason': value,
    };

BenefitForm _$BenefitFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('BenefitForm', json, ($checkedConvert) {
      final val = BenefitForm(
        levelName: $checkedConvert('levelName', (v) => v as String?),
        growthMin: $checkedConvert('growthMin', (v) => (v as num?)?.toInt()),
        growthMax: $checkedConvert('growthMax', (v) => (v as num?)?.toInt()),
        monthlyDehazeQuota: $checkedConvert(
          'monthlyDehazeQuota',
          (v) => (v as num?)?.toInt(),
        ),
        monthlyEvaluateQuota: $checkedConvert(
          'monthlyEvaluateQuota',
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
        sort: $checkedConvert('sort', (v) => (v as num?)?.toInt()),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$BenefitFormToJson(
  BenefitForm instance,
) => <String, dynamic>{
  if (instance.levelName case final value?) 'levelName': value,
  if (instance.growthMin case final value?) 'growthMin': value,
  if (instance.growthMax case final value?) 'growthMax': value,
  if (instance.monthlyDehazeQuota case final value?)
    'monthlyDehazeQuota': value,
  if (instance.monthlyEvaluateQuota case final value?)
    'monthlyEvaluateQuota': value,
  if (instance.historyRetention case final value?) 'historyRetention': value,
  if (instance.batchLimit case final value?) 'batchLimit': value,
  if (instance.priority case final value?) 'priority': value,
  if (instance.advancedParams case final value?) 'advancedParams': value,
  if (instance.hdExport case final value?) 'hdExport': value,
  if (instance.reportExport case final value?) 'reportExport': value,
  if (instance.batchDownload case final value?) 'batchDownload': value,
  if (instance.sort case final value?) 'sort': value,
  if (instance.status case final value?) 'status': value,
};

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

MemberProfileVO _$MemberProfileVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MemberProfileVO', json, ($checkedConvert) {
      final val = MemberProfileVO(
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        levelCode: $checkedConvert('levelCode', (v) => v as String),
        levelName: $checkedConvert('levelName', (v) => v as String),
        growthValue: $checkedConvert('growthValue', (v) => (v as num).toInt()),
        nextLevelGrowth: $checkedConvert(
          'nextLevelGrowth',
          (v) => (v as num?)?.toInt(),
        ),
        progressPercent: $checkedConvert(
          'progressPercent',
          (v) => (v as num).toInt(),
        ),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        monthlyDehazeQuota: $checkedConvert(
          'monthlyDehazeQuota',
          (v) => (v as num).toInt(),
        ),
        monthlyDehazeUsed: $checkedConvert(
          'monthlyDehazeUsed',
          (v) => (v as num).toInt(),
        ),
        monthlyEvaluateQuota: $checkedConvert(
          'monthlyEvaluateQuota',
          (v) => (v as num).toInt(),
        ),
        monthlyEvaluateUsed: $checkedConvert(
          'monthlyEvaluateUsed',
          (v) => (v as num).toInt(),
        ),
        benefits: $checkedConvert(
          'benefits',
          (v) => BenefitVO.fromJson(v as Map<String, dynamic>),
        ),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$MemberProfileVOToJson(MemberProfileVO instance) =>
    <String, dynamic>{
      'userId': instance.userId,
      'username': instance.username,
      'nickname': instance.nickname,
      if (instance.avatar case final value?) 'avatar': value,
      'levelCode': instance.levelCode,
      'levelName': instance.levelName,
      'growthValue': instance.growthValue,
      if (instance.nextLevelGrowth case final value?) 'nextLevelGrowth': value,
      'progressPercent': instance.progressPercent,
      if (instance.expireTime case final value?) 'expireTime': value,
      'monthlyDehazeQuota': instance.monthlyDehazeQuota,
      'monthlyDehazeUsed': instance.monthlyDehazeUsed,
      'monthlyEvaluateQuota': instance.monthlyEvaluateQuota,
      'monthlyEvaluateUsed': instance.monthlyEvaluateUsed,
      'benefits': instance.benefits.toJson(),
      'status': instance.status,
    };

MemberDetailVO _$MemberDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MemberDetailVO', json, ($checkedConvert) {
      final val = MemberDetailVO(
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        levelCode: $checkedConvert('levelCode', (v) => v as String),
        levelName: $checkedConvert('levelName', (v) => v as String),
        growthValue: $checkedConvert('growthValue', (v) => (v as num).toInt()),
        nextLevelGrowth: $checkedConvert(
          'nextLevelGrowth',
          (v) => (v as num?)?.toInt(),
        ),
        progressPercent: $checkedConvert(
          'progressPercent',
          (v) => (v as num).toInt(),
        ),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        monthlyDehazeQuota: $checkedConvert(
          'monthlyDehazeQuota',
          (v) => (v as num).toInt(),
        ),
        monthlyDehazeUsed: $checkedConvert(
          'monthlyDehazeUsed',
          (v) => (v as num).toInt(),
        ),
        monthlyEvaluateQuota: $checkedConvert(
          'monthlyEvaluateQuota',
          (v) => (v as num).toInt(),
        ),
        monthlyEvaluateUsed: $checkedConvert(
          'monthlyEvaluateUsed',
          (v) => (v as num).toInt(),
        ),
        benefits: $checkedConvert(
          'benefits',
          (v) => BenefitVO.fromJson(v as Map<String, dynamic>),
        ),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        levelSource: $checkedConvert('levelSource', (v) => v as String),
        totalConsumption: $checkedConvert(
          'totalConsumption',
          (v) => (v as num).toInt(),
        ),
        becomeMemberTime: $checkedConvert(
          'becomeMemberTime',
          (v) => v as String?,
        ),
        frozenReason: $checkedConvert('frozenReason', (v) => v as String?),
        frozenTime: $checkedConvert('frozenTime', (v) => v as String?),
        quotaResetMonth: $checkedConvert(
          'quotaResetMonth',
          (v) => (v as num?)?.toInt(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$MemberDetailVOToJson(
  MemberDetailVO instance,
) => <String, dynamic>{
  'userId': instance.userId,
  'username': instance.username,
  'nickname': instance.nickname,
  if (instance.avatar case final value?) 'avatar': value,
  'levelCode': instance.levelCode,
  'levelName': instance.levelName,
  'growthValue': instance.growthValue,
  if (instance.nextLevelGrowth case final value?) 'nextLevelGrowth': value,
  'progressPercent': instance.progressPercent,
  if (instance.expireTime case final value?) 'expireTime': value,
  'monthlyDehazeQuota': instance.monthlyDehazeQuota,
  'monthlyDehazeUsed': instance.monthlyDehazeUsed,
  'monthlyEvaluateQuota': instance.monthlyEvaluateQuota,
  'monthlyEvaluateUsed': instance.monthlyEvaluateUsed,
  'benefits': instance.benefits.toJson(),
  'status': instance.status,
  'levelSource': instance.levelSource,
  'totalConsumption': instance.totalConsumption,
  if (instance.becomeMemberTime case final value?) 'becomeMemberTime': value,
  if (instance.frozenReason case final value?) 'frozenReason': value,
  if (instance.frozenTime case final value?) 'frozenTime': value,
  if (instance.quotaResetMonth case final value?) 'quotaResetMonth': value,
};

MemberPageVO _$MemberPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MemberPageVO', json, ($checkedConvert) {
      final val = MemberPageVO(
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
        levelCode: $checkedConvert('levelCode', (v) => v as String),
        levelName: $checkedConvert('levelName', (v) => v as String),
        growthValue: $checkedConvert('growthValue', (v) => (v as num).toInt()),
        monthlyUsed: $checkedConvert('monthlyUsed', (v) => (v as num).toInt()),
        expireTime: $checkedConvert('expireTime', (v) => v as String?),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        becomeMemberTime: $checkedConvert(
          'becomeMemberTime',
          (v) => v as String?,
        ),
      );
      return val;
    });

Map<String, dynamic> _$MemberPageVOToJson(
  MemberPageVO instance,
) => <String, dynamic>{
  'userId': instance.userId,
  'username': instance.username,
  'nickname': instance.nickname,
  'levelCode': instance.levelCode,
  'levelName': instance.levelName,
  'growthValue': instance.growthValue,
  'monthlyUsed': instance.monthlyUsed,
  if (instance.expireTime case final value?) 'expireTime': value,
  'status': instance.status,
  if (instance.becomeMemberTime case final value?) 'becomeMemberTime': value,
};

GrowthLogVO _$GrowthLogVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('GrowthLogVO', json, ($checkedConvert) {
      final val = GrowthLogVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        changeType: $checkedConvert('changeType', (v) => v as String),
        changeValue: $checkedConvert('changeValue', (v) => (v as num).toInt()),
        balance: $checkedConvert('balance', (v) => (v as num).toInt()),
        relatedId: $checkedConvert('relatedId', (v) => v as String?),
        reason: $checkedConvert('reason', (v) => v as String?),
        operatorId: $checkedConvert('operatorId', (v) => (v as num?)?.toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$GrowthLogVOToJson(GrowthLogVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'changeType': instance.changeType,
      'changeValue': instance.changeValue,
      'balance': instance.balance,
      if (instance.relatedId case final value?) 'relatedId': value,
      if (instance.reason case final value?) 'reason': value,
      if (instance.operatorId case final value?) 'operatorId': value,
      'createTime': instance.createTime,
    };

SignInResultVO _$SignInResultVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('SignInResultVO', json, ($checkedConvert) {
      final val = SignInResultVO(
        signDate: $checkedConvert('signDate', (v) => v as String),
        continuousDays: $checkedConvert(
          'continuousDays',
          (v) => (v as num).toInt(),
        ),
        growthValue: $checkedConvert('growthValue', (v) => (v as num).toInt()),
        bonusGrowth: $checkedConvert('bonusGrowth', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$SignInResultVOToJson(SignInResultVO instance) =>
    <String, dynamic>{
      'signDate': instance.signDate,
      'continuousDays': instance.continuousDays,
      'growthValue': instance.growthValue,
      'bonusGrowth': instance.bonusGrowth,
    };

SignInCalendarVO _$SignInCalendarVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('SignInCalendarVO', json, ($checkedConvert) {
      final val = SignInCalendarVO(
        signDates: $checkedConvert(
          'signDates',
          (v) => (v as List<dynamic>).map((e) => e as String).toList(),
        ),
        continuousDays: $checkedConvert(
          'continuousDays',
          (v) => (v as num).toInt(),
        ),
        totalDays: $checkedConvert('totalDays', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$SignInCalendarVOToJson(SignInCalendarVO instance) =>
    <String, dynamic>{
      'signDates': instance.signDates,
      'continuousDays': instance.continuousDays,
      'totalDays': instance.totalDays,
    };
