import 'package:json_annotation/json_annotation.dart';

part 'announcement_model.g.dart';

// ==================== 公告 VO ====================

@JsonSerializable()
class AnnouncementVO {
  const AnnouncementVO({
    required this.id,
    required this.title,
    this.content,
    required this.type,
    this.typeName,
    required this.status,
    this.statusName,
    required this.priority,
    required this.targetType,
    this.targetUsers,
    this.sendTime,
    this.expireTime,
    required this.createTime,
    this.updateTime,
    this.createBy,
    this.read,
  });

  factory AnnouncementVO.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementVOFromJson(json);

  final int id;
  final String title;
  final String? content;
  final String type;
  final String? typeName;
  final String status;
  final String? statusName;
  final int priority;

  @JsonKey(name: 'targetType')
  final String targetType;

  @JsonKey(name: 'targetUsers')
  final List<int>? targetUsers;

  final String? sendTime;
  final String? expireTime;
  final String createTime;
  final String? updateTime;
  final int? createBy;
  final bool? read;

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
    this.startDate,
    this.endDate,
  });

  factory AnnouncementQuery.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? title;
  final String? type;
  final String? status;
  final String? startDate;
  final String? endDate;

  Map<String, dynamic> toJson() => _$AnnouncementQueryToJson(this);

  /// 转为 queryParameters 格式（仅非 null 字段）
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (title != null) 'title': title,
        if (type != null) 'type': type,
        if (status != null) 'status': status,
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      };
}

// ==================== 公告表单 ====================

@JsonSerializable()
class AnnouncementForm {
  const AnnouncementForm({
    this.id,
    required this.title,
    required this.content,
    required this.type,
    required this.priority,
    required this.targetType,
    this.targetUsers,
    this.expireTime,
    this.sendNow,
  });

  factory AnnouncementForm.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementFormFromJson(json);

  final int? id;
  final String title;
  final String content;
  final String type;
  final int priority;

  @JsonKey(name: 'targetType')
  final String targetType;

  @JsonKey(name: 'targetUsers')
  final List<int>? targetUsers;

  final String? expireTime;

  @JsonKey(name: 'sendNow')
  final bool? sendNow;

  Map<String, dynamic> toJson() => _$AnnouncementFormToJson(this);
}

// ==================== 公告发送结果 ====================

@JsonSerializable()
class AnnouncementSendResult {
  const AnnouncementSendResult({
    required this.successCount,
    required this.failedCount,
    required this.totalCount,
    required this.sendTime,
  });

  factory AnnouncementSendResult.fromJson(Map<String, dynamic> json) =>
      _$AnnouncementSendResultFromJson(json);

  @JsonKey(name: 'successCount')
  final int successCount;

  @JsonKey(name: 'failedCount')
  final int failedCount;

  @JsonKey(name: 'totalCount')
  final int totalCount;

  @JsonKey(name: 'sendTime')
  final String sendTime;

  Map<String, dynamic> toJson() => _$AnnouncementSendResultToJson(this);
}
