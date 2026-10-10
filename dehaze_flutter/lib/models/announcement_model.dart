import 'package:json_annotation/json_annotation.dart';

part 'announcement_model.g.dart';

/// 公告发送范围（后端取值小写：all/level/specified）
enum AnnouncementTargetScope {
  all('all', '全体用户'),
  level('level', '按会员等级'),
  specified('specified', '指定用户');

  const AnnouncementTargetScope(this.value, this.displayName);

  final String value;
  final String displayName;

  static AnnouncementTargetScope fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => AnnouncementTargetScope.all,
  );
}

AnnouncementTargetScope _targetScopeFromJson(dynamic value) =>
    AnnouncementTargetScope.fromValue(value is String ? value : null);

String _targetScopeToJson(AnnouncementTargetScope scope) => scope.value;

// ==================== 公告 VO ====================

/// 公告视图对象
///
/// 对应 JS SDK AnnouncementVO 与后端 AnnouncementVO / AnnouncementDetailVO（后者多 targetParams、updateTime）。
@JsonSerializable()
class AnnouncementVO {
  const AnnouncementVO({
    required this.id,
    required this.title,
    required this.type,
    required this.importance,
    required this.targetScope,
    required this.status,
    required this.createTime,
    this.content,
    this.typeLabel,
    this.importanceLabel,
    this.targetScopeLabel,
    this.targetParams,
    this.statusLabel,
    this.sendTime,
    this.expireTime,
    this.sentCount,
    this.updateTime,
    this.createBy,
  });

  factory AnnouncementVO.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementVOFromJson(json);

  final int id;
  final String title;
  final String? content;

  /// 公告类型
  final String type;
  final String? typeLabel;

  /// 重要级别（1 普通 / 2 重要）
  final int importance;
  final String? importanceLabel;

  /// 发送范围
  @JsonKey(fromJson: _targetScopeFromJson, toJson: _targetScopeToJson)
  final AnnouncementTargetScope targetScope;
  final String? targetScopeLabel;

  /// 发送范围参数（仅详情返回）
  final Map<String, dynamic>? targetParams;

  /// 公告状态（1 草稿 / 2 待发送 / 3 已发送 / 4 已取消）
  final int status;
  final String? statusLabel;

  final String? sendTime;
  final String? expireTime;

  /// 已发送人数
  final int? sentCount;

  final String createTime;
  final String? updateTime;
  final int? createBy;

  Map<String, dynamic> toJson() => _$AnnouncementVOToJson(this);
}

// ==================== 公告查询参数 ====================

@JsonSerializable()
class AnnouncementQuery {
  const AnnouncementQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.title,
    this.type,
    this.status,
  });

  factory AnnouncementQuery.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? title;
  final String? type;

  /// 公告状态（1 草稿 / 2 待发送 / 3 已发送 / 4 已取消）
  final int? status;

  Map<String, dynamic> toJson() => _$AnnouncementQueryToJson(this);

  /// 转为 queryParameters 格式（仅非 null 字段）
  Map<String, dynamic> toQuery() => {
    'pageNum': pageNum,
    'pageSize': pageSize,
    if (title != null) 'title': title,
    if (type != null) 'type': type,
    if (status != null) 'status': status,
  };
}

// ==================== 公告表单 ====================

@JsonSerializable()
class AnnouncementForm {
  const AnnouncementForm({
    required this.title,
    required this.content,
    required this.type,
    required this.importance,
    required this.targetScope,
    this.targetParams,
    this.sendTime,
    this.expireTime,
  });

  factory AnnouncementForm.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementFormFromJson(json);

  final String title;
  final String content;
  final String type;

  /// 重要级别（1 普通 / 2 重要）
  final int importance;

  /// 发送范围
  @JsonKey(fromJson: _targetScopeFromJson, toJson: _targetScopeToJson)
  final AnnouncementTargetScope targetScope;
  final Map<String, dynamic>? targetParams;
  final String? sendTime;
  final String? expireTime;

  Map<String, dynamic> toJson() => _$AnnouncementFormToJson(this);
}

// ==================== 公告发送结果 ====================

@JsonSerializable()
class AnnouncementSendResult {
  const AnnouncementSendResult({required this.sentCount});

  factory AnnouncementSendResult.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementSendResultFromJson(json);

  /// 已发送人数
  final int sentCount;

  Map<String, dynamic> toJson() => _$AnnouncementSendResultToJson(this);
}
