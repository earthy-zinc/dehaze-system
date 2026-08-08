import 'package:json_annotation/json_annotation.dart';

part 'member_model.g.dart';

// ==================== 枚举 ====================

/// 会员等级编码
enum MemberLevelCode {
  level0('level_0'),
  level1('level_1'),
  level2('level_2'),
  level3('level_3');

  const MemberLevelCode(this.value);
  final String value;
}

/// 等级来源
enum MemberLevelSource {
  growth('growth'),
  purchase('purchase'),
  admin('admin');

  const MemberLevelSource(this.value);
  final String value;
}

/// 成长值变动类型
enum GrowthChangeType {
  dehaze('dehaze'),
  evaluate('evaluate'),
  rating('rating'),
  signIn('sign_in'),
  signInBonus('sign_in_bonus'),
  consume('consume'),
  refundDeduct('refund_deduct'),
  adminAdjust('admin_adjust');

  const GrowthChangeType(this.value);
  final String value;
}

/// 会员状态（1:正常;0:冻结）
enum MemberStatus {
  active(1),
  frozen(0);

  const MemberStatus(this.value);
  final int value;
}

// ==================== 模型 ====================

/// 会员查询参数
@JsonSerializable()
class MemberQuery {
  const MemberQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keywords,
    this.levelCode,
    this.status,
    this.expireTimeStart,
    this.expireTimeEnd,
    this.growthMin,
    this.growthMax,
  });

  factory MemberQuery.fromJson(Map<String, dynamic> json) =>
      _$MemberQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? keywords;

  @JsonKey(name: 'levelCode')
  final String? levelCode;

  @JsonKey(name: 'status')
  final int? status;

  final String? expireTimeStart;
  final String? expireTimeEnd;
  final int? growthMin;
  final int? growthMax;

  Map<String, dynamic> toJson() => _$MemberQueryToJson(this);
}

/// 成长值流水查询参数
@JsonSerializable()
class GrowthLogQuery {
  const GrowthLogQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.changeType,
    this.startTime,
    this.endTime,
  });

  factory GrowthLogQuery.fromJson(Map<String, dynamic> json) =>
      _$GrowthLogQueryFromJson(json);

  final int pageNum;
  final int pageSize;

  @JsonKey(name: 'changeType')
  final String? changeType;

  final String? startTime;
  final String? endTime;

  Map<String, dynamic> toJson() => _$GrowthLogQueryToJson(this);
}

/// 等级调整表单
@JsonSerializable()
class MemberLevelAdjustForm {
  const MemberLevelAdjustForm({
    required this.levelCode,
    this.expireTime,
    required this.reason,
  });

  factory MemberLevelAdjustForm.fromJson(Map<String, dynamic> json) =>
      _$MemberLevelAdjustFormFromJson(json);

  final String levelCode;
  final String? expireTime;
  final String reason;

  Map<String, dynamic> toJson() => _$MemberLevelAdjustFormToJson(this);
}

/// 成长值调整表单
@JsonSerializable()
class MemberGrowthAdjustForm {
  const MemberGrowthAdjustForm({
    required this.changeValue,
    required this.reason,
  });

  factory MemberGrowthAdjustForm.fromJson(Map<String, dynamic> json) =>
      _$MemberGrowthAdjustFormFromJson(json);

  final int changeValue;
  final String reason;

  Map<String, dynamic> toJson() => _$MemberGrowthAdjustFormToJson(this);
}

/// 会员状态变更表单
@JsonSerializable()
class MemberStatusForm {
  const MemberStatusForm({
    required this.status,
    this.reason,
  });

  factory MemberStatusForm.fromJson(Map<String, dynamic> json) =>
      _$MemberStatusFormFromJson(json);

  final int status;
  final String? reason;

  Map<String, dynamic> toJson() => _$MemberStatusFormToJson(this);
}

/// 权益配置表单
@JsonSerializable()
class BenefitForm {
  const BenefitForm({
    this.levelName,
    this.growthMin,
    this.growthMax,
    this.monthlyDehazeQuota,
    this.monthlyEvaluateQuota,
    this.historyRetention,
    this.batchLimit,
    this.priority,
    this.advancedParams,
    this.hdExport,
    this.reportExport,
    this.batchDownload,
    this.sort,
    this.status,
  });

  factory BenefitForm.fromJson(Map<String, dynamic> json) =>
      _$BenefitFormFromJson(json);

  final String? levelName;
  final int? growthMin;
  final int? growthMax;
  final int? monthlyDehazeQuota;
  final int? monthlyEvaluateQuota;
  final int? historyRetention;
  final int? batchLimit;
  final int? priority;
  final int? advancedParams;
  final int? hdExport;
  final int? reportExport;
  final int? batchDownload;
  final int? sort;
  final int? status;

  Map<String, dynamic> toJson() => _$BenefitFormToJson(this);
}

/// 权益配置VO
@JsonSerializable()
class BenefitVO {
  const BenefitVO({
    required this.levelCode,
    required this.levelName,
    required this.growthMin,
    required this.growthMax,
    required this.monthlyDehazeQuota,
    required this.monthlyEvaluateQuota,
    required this.historyRetention,
    required this.batchLimit,
    required this.priority,
    required this.advancedParams,
    required this.hdExport,
    required this.reportExport,
    required this.batchDownload,
    required this.sort,
    required this.status,
  });

  factory BenefitVO.fromJson(Map<String, dynamic> json) =>
      _$BenefitVOFromJson(json);

  final String levelCode;
  final String levelName;
  final int growthMin;
  final int growthMax;
  final int monthlyDehazeQuota;
  final int monthlyEvaluateQuota;
  final int historyRetention;
  final int batchLimit;
  final int priority;
  final int advancedParams;
  final int hdExport;
  final int reportExport;
  final int batchDownload;
  final int sort;
  final int status;

  Map<String, dynamic> toJson() => _$BenefitVOToJson(this);
}

/// 会员信息VO（用户端profile）
@JsonSerializable()
class MemberProfileVO {
  const MemberProfileVO({
    required this.userId,
    required this.username,
    required this.nickname,
    this.avatar,
    required this.levelCode,
    required this.levelName,
    required this.growthValue,
    this.nextLevelGrowth,
    required this.progressPercent,
    this.expireTime,
    required this.monthlyDehazeQuota,
    required this.monthlyDehazeUsed,
    required this.monthlyEvaluateQuota,
    required this.monthlyEvaluateUsed,
    required this.benefits,
    required this.status,
  });

  factory MemberProfileVO.fromJson(Map<String, dynamic> json) =>
      _$MemberProfileVOFromJson(json);

  final int userId;
  final String username;
  final String nickname;
  final String? avatar;
  final String levelCode;
  final String levelName;
  final int growthValue;
  final int? nextLevelGrowth;
  final int progressPercent;
  final String? expireTime;
  final int monthlyDehazeQuota;
  final int monthlyDehazeUsed;
  final int monthlyEvaluateQuota;
  final int monthlyEvaluateUsed;
  final BenefitVO benefits;
  final int status;

  Map<String, dynamic> toJson() => _$MemberProfileVOToJson(this);
}

/// 会员详情VO（后台），继承自 MemberProfileVO 的字段
@JsonSerializable()
class MemberDetailVO {
  const MemberDetailVO({
    required this.userId,
    required this.username,
    required this.nickname,
    this.avatar,
    required this.levelCode,
    required this.levelName,
    required this.growthValue,
    this.nextLevelGrowth,
    required this.progressPercent,
    this.expireTime,
    required this.monthlyDehazeQuota,
    required this.monthlyDehazeUsed,
    required this.monthlyEvaluateQuota,
    required this.monthlyEvaluateUsed,
    required this.benefits,
    required this.status,
    required this.levelSource,
    required this.totalConsumption,
    this.becomeMemberTime,
    this.frozenReason,
    this.frozenTime,
    this.quotaResetMonth,
  });

  factory MemberDetailVO.fromJson(Map<String, dynamic> json) =>
      _$MemberDetailVOFromJson(json);

  // —— 同 MemberProfileVO ——
  final int userId;
  final String username;
  final String nickname;
  final String? avatar;
  final String levelCode;
  final String levelName;
  final int growthValue;
  final int? nextLevelGrowth;
  final int progressPercent;
  final String? expireTime;
  final int monthlyDehazeQuota;
  final int monthlyDehazeUsed;
  final int monthlyEvaluateQuota;
  final int monthlyEvaluateUsed;
  final BenefitVO benefits;
  final int status;

  // —— 额外管理端字段 ——
  final String levelSource;
  final int totalConsumption;
  final String? becomeMemberTime;
  final String? frozenReason;
  final String? frozenTime;
  final int? quotaResetMonth;

  Map<String, dynamic> toJson() => _$MemberDetailVOToJson(this);
}

/// 会员分页VO
@JsonSerializable()
class MemberPageVO {
  const MemberPageVO({
    required this.userId,
    required this.username,
    required this.nickname,
    required this.levelCode,
    required this.levelName,
    required this.growthValue,
    required this.monthlyUsed,
    this.expireTime,
    required this.status,
    this.becomeMemberTime,
  });

  factory MemberPageVO.fromJson(Map<String, dynamic> json) =>
      _$MemberPageVOFromJson(json);

  final int userId;
  final String username;
  final String nickname;
  final String levelCode;
  final String levelName;
  final int growthValue;
  final int monthlyUsed;
  final String? expireTime;
  final int status;
  final String? becomeMemberTime;

  Map<String, dynamic> toJson() => _$MemberPageVOToJson(this);
}

/// 成长值流水VO
@JsonSerializable()
class GrowthLogVO {
  const GrowthLogVO({
    required this.id,
    required this.changeType,
    required this.changeValue,
    required this.balance,
    this.relatedId,
    this.reason,
    this.operatorId,
    required this.createTime,
  });

  factory GrowthLogVO.fromJson(Map<String, dynamic> json) =>
      _$GrowthLogVOFromJson(json);

  final int id;
  final String changeType;
  final int changeValue;
  final int balance;
  final String? relatedId;
  final String? reason;
  final int? operatorId;
  final String createTime;

  Map<String, dynamic> toJson() => _$GrowthLogVOToJson(this);
}

/// 签到结果VO
@JsonSerializable()
class SignInResultVO {
  const SignInResultVO({
    required this.signDate,
    required this.continuousDays,
    required this.growthValue,
    required this.bonusGrowth,
  });

  factory SignInResultVO.fromJson(Map<String, dynamic> json) =>
      _$SignInResultVOFromJson(json);

  final String signDate;
  final int continuousDays;
  final int growthValue;
  final int bonusGrowth;

  Map<String, dynamic> toJson() => _$SignInResultVOToJson(this);
}

/// 签到日历VO
@JsonSerializable()
class SignInCalendarVO {
  const SignInCalendarVO({
    required this.signDates,
    required this.continuousDays,
    required this.totalDays,
  });

  factory SignInCalendarVO.fromJson(Map<String, dynamic> json) =>
      _$SignInCalendarVOFromJson(json);

  final List<String> signDates;
  final int continuousDays;
  final int totalDays;

  Map<String, dynamic> toJson() => _$SignInCalendarVOToJson(this);
}
