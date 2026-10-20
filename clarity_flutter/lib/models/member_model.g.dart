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
      'keywords': ?instance.keywords,
      'levelCode': ?instance.levelCode,
      'status': ?instance.status,
      'expireTimeStart': ?instance.expireTimeStart,
      'expireTimeEnd': ?instance.expireTimeEnd,
      'growthMin': ?instance.growthMin,
      'growthMax': ?instance.growthMax,
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
      'changeType': ?instance.changeType,
      'startTime': ?instance.startTime,
      'endTime': ?instance.endTime,
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
  'expireTime': ?instance.expireTime,
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
    <String, dynamic>{'status': instance.status, 'reason': ?instance.reason};

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
        multimodalLimit: $checkedConvert(
          'multimodalLimit',
          (v) => (v as num?)?.toInt(),
        ),
        maxDevices: $checkedConvert('maxDevices', (v) => (v as num?)?.toInt()),
        vipGiftCredits: $checkedConvert(
          'vipGiftCredits',
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

Map<String, dynamic> _$BenefitFormToJson(BenefitForm instance) =>
    <String, dynamic>{
      'levelName': ?instance.levelName,
      'growthMin': ?instance.growthMin,
      'growthMax': ?instance.growthMax,
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
      'multimodalLimit': ?instance.multimodalLimit,
      'maxDevices': ?instance.maxDevices,
      'vipGiftCredits': ?instance.vipGiftCredits,
      'historyRetention': ?instance.historyRetention,
      'batchLimit': ?instance.batchLimit,
      'priority': ?instance.priority,
      'advancedParams': ?instance.advancedParams,
      'hdExport': ?instance.hdExport,
      'reportExport': ?instance.reportExport,
      'batchDownload': ?instance.batchDownload,
      'sort': ?instance.sort,
      'status': ?instance.status,
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
    monthlyDerainQuota: $checkedConvert(
      'monthlyDerainQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyDesnowQuota: $checkedConvert(
      'monthlyDesnowQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyLowlightQuota: $checkedConvert(
      'monthlyLowlightQuota',
      (v) => (v as num).toInt(),
    ),
    monthlySuperResolutionQuota: $checkedConvert(
      'monthlySuperResolutionQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyDenoiseQuota: $checkedConvert(
      'monthlyDenoiseQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyInpaintQuota: $checkedConvert(
      'monthlyInpaintQuota',
      (v) => (v as num).toInt(),
    ),
    monthlyEvaluateQuota: $checkedConvert(
      'monthlyEvaluateQuota',
      (v) => (v as num).toInt(),
    ),
    aiCreditsDaily: $checkedConvert(
      'aiCreditsDaily',
      (v) => (v as num).toInt(),
    ),
    aiCreditsMonthly: $checkedConvert(
      'aiCreditsMonthly',
      (v) => (v as num).toInt(),
    ),
    multimodalLimit: $checkedConvert(
      'multimodalLimit',
      (v) => (v as num).toInt(),
    ),
    maxDevices: $checkedConvert('maxDevices', (v) => (v as num).toInt()),
    vipGiftCredits: $checkedConvert(
      'vipGiftCredits',
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
  'monthlyDerainQuota': instance.monthlyDerainQuota,
  'monthlyDesnowQuota': instance.monthlyDesnowQuota,
  'monthlyLowlightQuota': instance.monthlyLowlightQuota,
  'monthlySuperResolutionQuota': instance.monthlySuperResolutionQuota,
  'monthlyDenoiseQuota': instance.monthlyDenoiseQuota,
  'monthlyInpaintQuota': instance.monthlyInpaintQuota,
  'monthlyEvaluateQuota': instance.monthlyEvaluateQuota,
  'aiCreditsDaily': instance.aiCreditsDaily,
  'aiCreditsMonthly': instance.aiCreditsMonthly,
  'multimodalLimit': instance.multimodalLimit,
  'maxDevices': instance.maxDevices,
  'vipGiftCredits': instance.vipGiftCredits,
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
        levelSource: $checkedConvert('levelSource', (v) => v as String?),
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
      'avatar': ?instance.avatar,
      'levelCode': instance.levelCode,
      'levelName': instance.levelName,
      'levelSource': ?instance.levelSource,
      'growthValue': instance.growthValue,
      'nextLevelGrowth': ?instance.nextLevelGrowth,
      'progressPercent': instance.progressPercent,
      'expireTime': ?instance.expireTime,
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

Map<String, dynamic> _$MemberDetailVOToJson(MemberDetailVO instance) =>
    <String, dynamic>{
      'userId': instance.userId,
      'username': instance.username,
      'nickname': instance.nickname,
      'avatar': ?instance.avatar,
      'levelCode': instance.levelCode,
      'levelName': instance.levelName,
      'growthValue': instance.growthValue,
      'nextLevelGrowth': ?instance.nextLevelGrowth,
      'progressPercent': instance.progressPercent,
      'expireTime': ?instance.expireTime,
      'monthlyDehazeQuota': instance.monthlyDehazeQuota,
      'monthlyDehazeUsed': instance.monthlyDehazeUsed,
      'monthlyEvaluateQuota': instance.monthlyEvaluateQuota,
      'monthlyEvaluateUsed': instance.monthlyEvaluateUsed,
      'benefits': instance.benefits.toJson(),
      'status': instance.status,
      'levelSource': instance.levelSource,
      'totalConsumption': instance.totalConsumption,
      'becomeMemberTime': ?instance.becomeMemberTime,
      'frozenReason': ?instance.frozenReason,
      'frozenTime': ?instance.frozenTime,
      'quotaResetMonth': ?instance.quotaResetMonth,
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

Map<String, dynamic> _$MemberPageVOToJson(MemberPageVO instance) =>
    <String, dynamic>{
      'userId': instance.userId,
      'username': instance.username,
      'nickname': instance.nickname,
      'levelCode': instance.levelCode,
      'levelName': instance.levelName,
      'growthValue': instance.growthValue,
      'monthlyUsed': instance.monthlyUsed,
      'expireTime': ?instance.expireTime,
      'status': instance.status,
      'becomeMemberTime': ?instance.becomeMemberTime,
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
      'relatedId': ?instance.relatedId,
      'reason': ?instance.reason,
      'operatorId': ?instance.operatorId,
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
