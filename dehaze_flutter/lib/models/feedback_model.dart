import 'package:json_annotation/json_annotation.dart';

part 'feedback_model.g.dart';

// ==================== 枚举 ====================

/// 反馈状态
enum FeedbackStatus {
  @JsonValue(0)
  pending,
  @JsonValue(1)
  processing,
  @JsonValue(2)
  resolved,
  @JsonValue(3)
  closed,
  @JsonValue(4)
  rejected;

  int get value => switch (this) {
        FeedbackStatus.pending => 0,
        FeedbackStatus.processing => 1,
        FeedbackStatus.resolved => 2,
        FeedbackStatus.closed => 3,
        FeedbackStatus.rejected => 4,
      };

  static FeedbackStatus fromValue(int? value) {
    switch (value) {
      case 1:
        return FeedbackStatus.processing;
      case 2:
        return FeedbackStatus.resolved;
      case 3:
        return FeedbackStatus.closed;
      case 4:
        return FeedbackStatus.rejected;
      default:
        return FeedbackStatus.pending;
    }
  }
}

/// 反馈类型
enum FeedbackType {
  @JsonValue('bug')
  bug,
  @JsonValue('suggestion')
  suggestion,
  @JsonValue('question')
  question,
  @JsonValue('other')
  other;

  String get value => switch (this) {
        FeedbackType.bug => 'bug',
        FeedbackType.suggestion => 'suggestion',
        FeedbackType.question => 'question',
        FeedbackType.other => 'other',
      };

  static FeedbackType fromValue(String? value) {
    switch (value) {
      case 'suggestion':
        return FeedbackType.suggestion;
      case 'question':
        return FeedbackType.question;
      case 'other':
        return FeedbackType.other;
      default:
        return FeedbackType.bug;
    }
  }
}

/// 反馈回复类型
enum FeedbackReplyType {
  @JsonValue('reply')
  reply,
  @JsonValue('supplement')
  supplement,
  @JsonValue('system')
  system;

  String get value => switch (this) {
        FeedbackReplyType.reply => 'reply',
        FeedbackReplyType.supplement => 'supplement',
        FeedbackReplyType.system => 'system',
      };

  static FeedbackReplyType fromValue(String? value) {
    switch (value) {
      case 'supplement':
        return FeedbackReplyType.supplement;
      case 'system':
        return FeedbackReplyType.system;
      default:
        return FeedbackReplyType.reply;
    }
  }
}

/// 回复人类型
enum ReplierType {
  @JsonValue('user')
  user,
  @JsonValue('admin')
  admin,
  @JsonValue('system')
  system;

  String get value => switch (this) {
        ReplierType.user => 'user',
        ReplierType.admin => 'admin',
        ReplierType.system => 'system',
      };

  static ReplierType fromValue(String? value) {
    switch (value) {
      case 'admin':
        return ReplierType.admin;
      case 'system':
        return ReplierType.system;
      default:
        return ReplierType.user;
    }
  }
}

// ==================== 评分模型 ====================

/// 创建评分表单
@JsonSerializable()
class RatingCreateForm {
  const RatingCreateForm({
    required this.predictionLogId,
    required this.algorithmId,
    required this.rating,
    this.comment,
  });

  factory RatingCreateForm.fromJson(Map<String, dynamic> json) =>
      _$RatingCreateFormFromJson(json);

  /// 预测日志 ID
  final int predictionLogId;

  /// 算法 ID
  final int algorithmId;

  /// 评分（1-5）
  final int rating;

  /// 评价内容
  final String? comment;

  Map<String, dynamic> toJson() => _$RatingCreateFormToJson(this);
}

/// 评分查询参数
@JsonSerializable()
class RatingQuery {
  const RatingQuery({
    this.algorithmId,
    this.userId,
    this.pageNum = 1,
    this.pageSize = 10,
    this.minRating,
    this.maxRating,
    this.sortBy,
    this.sortOrder,
  });

  factory RatingQuery.fromJson(Map<String, dynamic> json) =>
      _$RatingQueryFromJson(json);

  /// 算法 ID
  final int? algorithmId;

  /// 用户 ID
  final int? userId;

  /// 页码
  final int pageNum;

  /// 每页条数
  final int pageSize;

  /// 最低评分
  final int? minRating;

  /// 最高评分
  final int? maxRating;

  /// 排序字段
  final String? sortBy;

  /// 排序方向（asc/desc）
  final String? sortOrder;

  Map<String, dynamic> toJson() => _$RatingQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (algorithmId != null) 'algorithmId': algorithmId,
        if (userId != null) 'userId': userId,
        if (minRating != null) 'minRating': minRating,
        if (maxRating != null) 'maxRating': maxRating,
        if (sortBy != null) 'sortBy': sortBy,
        if (sortOrder != null) 'sortOrder': sortOrder,
      };
}

/// 我的评分 VO（用户端）
@JsonSerializable()
class MyRatingVO {
  const MyRatingVO({
    required this.id,
    required this.predictionLogId,
    required this.algorithmId,
    required this.algorithmName,
    required this.rating,
    this.comment,
    required this.createTime,
  });

  factory MyRatingVO.fromJson(Map<String, dynamic> json) =>
      _$MyRatingVOFromJson(json);

  final int id;
  final int predictionLogId;
  final int algorithmId;
  final String algorithmName;
  final int rating;
  final String? comment;
  final String createTime;

  Map<String, dynamic> toJson() => _$MyRatingVOToJson(this);
}

/// 评分分页项 VO（后台）
@JsonSerializable()
class RatingPageVO {
  const RatingPageVO({
    required this.id,
    required this.userId,
    this.username,
    this.nickname,
    this.avatar,
    required this.algorithmId,
    required this.algorithmName,
    required this.predictionLogId,
    required this.rating,
    this.comment,
    required this.createTime,
  });

  factory RatingPageVO.fromJson(Map<String, dynamic> json) =>
      _$RatingPageVOFromJson(json);

  final int id;
  final int userId;
  final String? username;
  final String? nickname;
  final String? avatar;
  final int algorithmId;
  final String algorithmName;
  final int predictionLogId;
  final int rating;
  final String? comment;
  final String createTime;

  Map<String, dynamic> toJson() => _$RatingPageVOToJson(this);
}

/// 评分详情 VO
@JsonSerializable()
class RatingDetailVO {
  const RatingDetailVO({
    required this.id,
    required this.userId,
    this.username,
    this.nickname,
    this.avatar,
    required this.algorithmId,
    required this.algorithmName,
    required this.predictionLogId,
    required this.rating,
    this.comment,
    required this.createTime,
    this.predictionUrl,
    this.resultUrl,
    this.feedback,
  });

  factory RatingDetailVO.fromJson(Map<String, dynamic> json) =>
      _$RatingDetailVOFromJson(json);

  final int id;
  final int userId;
  final String? username;
  final String? nickname;
  final String? avatar;
  final int algorithmId;
  final String algorithmName;
  final int predictionLogId;
  final int rating;
  final String? comment;
  final String createTime;
  final String? predictionUrl;
  final String? resultUrl;
  final String? feedback;

  Map<String, dynamic> toJson() => _$RatingDetailVOToJson(this);
}

/// 评分统计 VO
@JsonSerializable()
class RatingStatsVO {
  const RatingStatsVO({
    required this.totalRatings,
    required this.averageRating,
    required this.fiveStar,
    required this.fourStar,
    required this.threeStar,
    required this.twoStar,
    required this.oneStar,
    required this.distribution,
  });

  factory RatingStatsVO.fromJson(Map<String, dynamic> json) =>
      _$RatingStatsVOFromJson(json);

  final int totalRatings;
  final double averageRating;
  final int fiveStar;
  final int fourStar;
  final int threeStar;
  final int twoStar;
  final int oneStar;

  /// 评分分布（key: 星级字符串 "1"-"5", value: 数量）
  final Map<String, int> distribution;

  Map<String, dynamic> toJson() => _$RatingStatsVOToJson(this);
}

// ==================== 反馈模型 ====================

/// 创建反馈表单
@JsonSerializable()
class FeedbackCreateForm {
  const FeedbackCreateForm({
    required this.type,
    required this.title,
    required this.content,
    this.contact,
    this.images,
  });

  factory FeedbackCreateForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackCreateFormFromJson(json);

  /// 反馈类型
  final String type;

  /// 标题
  final String title;

  /// 内容
  final String content;

  /// 联系方式
  final String? contact;

  /// 图片列表
  final List<String>? images;

  Map<String, dynamic> toJson() => _$FeedbackCreateFormToJson(this);
}

/// 反馈查询参数
@JsonSerializable()
class FeedbackQuery {
  const FeedbackQuery({
    this.type,
    this.status,
    this.keyword,
    this.pageNum = 1,
    this.pageSize = 10,
    this.userId,
    this.startDate,
    this.endDate,
  });

  factory FeedbackQuery.fromJson(Map<String, dynamic> json) =>
      _$FeedbackQueryFromJson(json);

  /// 反馈类型
  final String? type;

  /// 状态
  final String? status;

  /// 关键词
  final String? keyword;

  /// 页码
  final int pageNum;

  /// 每页条数
  final int pageSize;

  /// 用户 ID
  final int? userId;

  /// 开始日期
  final String? startDate;

  /// 结束日期
  final String? endDate;

  Map<String, dynamic> toJson() => _$FeedbackQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (type != null) 'type': type,
        if (status != null) 'status': status,
        if (keyword != null) 'keyword': keyword,
        if (userId != null) 'userId': userId,
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      };
}

/// 反馈分页项 VO
@JsonSerializable()
class FeedbackPageVO {
  const FeedbackPageVO({
    required this.id,
    required this.userId,
    this.username,
    this.nickname,
    required this.type,
    this.typeName,
    required this.title,
    required this.status,
    this.statusName,
    required this.createTime,
    this.updateTime,
  });

  factory FeedbackPageVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackPageVOFromJson(json);

  final int id;
  final int userId;
  final String? username;
  final String? nickname;
  final String type;
  final String? typeName;
  final String title;
  final String status;
  final String? statusName;
  final String createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$FeedbackPageVOToJson(this);
}

/// 反馈详情 VO
@JsonSerializable()
class FeedbackDetailVO {
  const FeedbackDetailVO({
    required this.id,
    required this.userId,
    this.username,
    this.nickname,
    this.avatar,
    required this.type,
    this.typeName,
    required this.title,
    required this.content,
    this.contact,
    this.images = const [],
    required this.status,
    this.statusName,
    required this.createTime,
    this.updateTime,
    this.replies,
  });

  factory FeedbackDetailVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackDetailVOFromJson(json);

  final int id;
  final int userId;
  final String? username;
  final String? nickname;
  final String? avatar;
  final String type;
  final String? typeName;
  final String title;
  final String content;
  final String? contact;
  final List<String> images;
  final String status;
  final String? statusName;
  final String createTime;
  final String? updateTime;
  final List<FeedbackReplyVO>? replies;

  Map<String, dynamic> toJson() => _$FeedbackDetailVOToJson(this);
}

/// 反馈回复 VO
@JsonSerializable()
class FeedbackReplyVO {
  const FeedbackReplyVO({
    required this.id,
    required this.feedbackId,
    required this.content,
    this.replyType,
    this.replierType,
    this.replierId,
    this.replierName,
    required this.createTime,
  });

  factory FeedbackReplyVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackReplyVOFromJson(json);

  final int id;
  final int feedbackId;
  final String content;
  final String? replyType;
  final String? replierType;
  final int? replierId;
  final String? replierName;
  final String createTime;

  Map<String, dynamic> toJson() => _$FeedbackReplyVOToJson(this);
}

/// 补充反馈表单
@JsonSerializable()
class FeedbackSupplementForm {
  const FeedbackSupplementForm({
    required this.feedbackId,
    required this.content,
    this.images,
  });

  factory FeedbackSupplementForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackSupplementFormFromJson(json);

  final int feedbackId;
  final String content;
  final List<String>? images;

  Map<String, dynamic> toJson() => _$FeedbackSupplementFormToJson(this);
}

/// 回复反馈表单
@JsonSerializable()
class FeedbackReplyForm {
  const FeedbackReplyForm({
    required this.feedbackId,
    required this.content,
  });

  factory FeedbackReplyForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackReplyFormFromJson(json);

  final int feedbackId;
  final String content;

  Map<String, dynamic> toJson() => _$FeedbackReplyFormToJson(this);
}

/// 分配反馈表单
@JsonSerializable()
class FeedbackAssignForm {
  const FeedbackAssignForm({
    required this.feedbackId,
    required this.assigneeId,
  });

  factory FeedbackAssignForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackAssignFormFromJson(json);

  final int feedbackId;
  final int assigneeId;

  Map<String, dynamic> toJson() => _$FeedbackAssignFormToJson(this);
}

/// 关闭反馈表单
@JsonSerializable()
class FeedbackCloseForm {
  const FeedbackCloseForm({
    required this.feedbackId,
    required this.reason,
  });

  factory FeedbackCloseForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackCloseFormFromJson(json);

  final int feedbackId;
  final String reason;

  Map<String, dynamic> toJson() => _$FeedbackCloseFormToJson(this);
}

/// 反馈统计 VO
@JsonSerializable()
class FeedbackStatsVO {
  const FeedbackStatsVO({
    required this.totalFeedbacks,
    required this.pendingFeedbacks,
    required this.resolvedFeedbacks,
    required this.closedFeedbacks,
    required this.typeDistribution,
    required this.statusDistribution,
  });

  factory FeedbackStatsVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackStatsVOFromJson(json);

  final int totalFeedbacks;
  final int pendingFeedbacks;
  final int resolvedFeedbacks;
  final int closedFeedbacks;

  /// 类型分布（key: 类型字符串, value: 数量）
  final Map<String, int> typeDistribution;

  /// 状态分布（key: 状态字符串, value: 数量）
  final Map<String, int> statusDistribution;

  Map<String, dynamic> toJson() => _$FeedbackStatsVOToJson(this);
}
