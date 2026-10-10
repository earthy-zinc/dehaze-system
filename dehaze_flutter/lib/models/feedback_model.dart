import 'package:json_annotation/json_annotation.dart';

part 'feedback_model.g.dart';

// ==================== 枚举 ====================

/// 反馈状态（pending 待处理 / processing 处理中 / replied 已回复 / closed 已关闭）
enum FeedbackStatus {
  pending('pending', '待处理'),
  processing('processing', '处理中'),
  replied('replied', '已回复'),
  closed('closed', '已关闭');

  const FeedbackStatus(this.value, this.displayName);

  final String value;
  final String displayName;

  static FeedbackStatus fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => FeedbackStatus.pending,
  );
}

/// 反馈类型
enum FeedbackType {
  suggestion('suggestion', '功能建议'),
  bug('bug', '问题报告'),
  experience('experience', '体验反馈'),
  complaint('complaint', '投诉');

  const FeedbackType(this.value, this.displayName);

  final String value;
  final String displayName;

  static FeedbackType fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => FeedbackType.suggestion,
  );
}

/// 回复类型
enum FeedbackReplyType {
  info('info', '信息补充'),
  resolved('resolved', '已解决'),
  unsupported('unsupported', '暂不支持'),
  devTransfer('dev_transfer', '转开发');

  const FeedbackReplyType(this.value, this.displayName);

  final String value;
  final String displayName;

  static FeedbackReplyType fromValue(String? value) => values.firstWhere(
    (e) => e.value == value,
    orElse: () => FeedbackReplyType.info,
  );
}

/// 回复人类型（1 用户 / 2 管理员）
enum ReplierType {
  user(1, '用户'),
  admin(2, '管理员');

  const ReplierType(this.value, this.displayName);

  final int value;
  final String displayName;

  static ReplierType fromValue(int? value) =>
      value == 2 ? ReplierType.admin : ReplierType.user;
}

// ==================== JSON 转换辅助函数 ====================

FeedbackStatus _feedbackStatusFromJson(dynamic value) =>
    FeedbackStatus.fromValue(value is String ? value : null);

String _feedbackStatusToJson(FeedbackStatus status) => status.value;

FeedbackType _feedbackTypeFromJson(dynamic value) =>
    FeedbackType.fromValue(value is String ? value : null);

String _feedbackTypeToJson(FeedbackType type) => type.value;

FeedbackReplyType? _feedbackReplyTypeFromJson(dynamic value) =>
    value is String ? FeedbackReplyType.fromValue(value) : null;

String? _feedbackReplyTypeToJson(FeedbackReplyType? type) => type?.value;

ReplierType _replierTypeFromJson(dynamic value) =>
    ReplierType.fromValue(value is num ? value.toInt() : null);

int _replierTypeToJson(ReplierType type) => type.value;

// ==================== 评分模型 ====================

/// 创建评分表单
@JsonSerializable()
class RatingCreateForm {
  const RatingCreateForm({
    required this.predLogId,
    required this.rating,
    this.comment,
  });

  factory RatingCreateForm.fromJson(Map<String, dynamic> json) =>
      _$RatingCreateFormFromJson(json);

  /// 图像任务 ID
  final int predLogId;

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
    this.modelId,
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

  /// 图像模型标识（统一模型规范 slug）
  final String? modelId;

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
    if (modelId != null) 'modelId': modelId,
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
    required this.predLogId,
    required this.modelName,
    required this.rating,
    this.comment,
    required this.createTime,
  });

  factory MyRatingVO.fromJson(Map<String, dynamic> json) =>
      _$MyRatingVOFromJson(json);

  final int id;
  final int predLogId;
  final String modelName;
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
    required this.modelName,
    required this.predLogId,
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
  final String modelName;
  final int predLogId;
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
    required this.modelId,
    required this.modelName,
    required this.predLogId,
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

  /// 图像模型标识（统一模型规范 slug）
  final String modelId;
  final String modelName;
  final int predLogId;
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
    required this.feedbackType,
    required this.title,
    required this.content,
    this.contact,
    this.images,
    this.relatedModule,
  });

  factory FeedbackCreateForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackCreateFormFromJson(json);

  /// 反馈类型
  @JsonKey(fromJson: _feedbackTypeFromJson, toJson: _feedbackTypeToJson)
  final FeedbackType feedbackType;

  /// 标题
  final String title;

  /// 内容
  final String content;

  /// 联系方式
  final String? contact;

  /// 图片列表
  final List<String>? images;

  /// 关联模块
  final String? relatedModule;

  Map<String, dynamic> toJson() => _$FeedbackCreateFormToJson(this);
}

/// 反馈查询参数（后台）
@JsonSerializable()
class FeedbackQuery {
  const FeedbackQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keywords,
    this.feedbackType,
    this.status,
    this.relatedModule,
    this.priority,
    this.assigneeId,
    this.startTime,
    this.endTime,
  });

  factory FeedbackQuery.fromJson(Map<String, dynamic> json) =>
      _$FeedbackQueryFromJson(json);

  /// 页码
  final int pageNum;

  /// 每页条数
  final int pageSize;

  /// 关键词
  final String? keywords;

  /// 反馈类型
  final String? feedbackType;

  /// 状态（pending/processing/replied/closed）
  final String? status;

  /// 关联模块
  final String? relatedModule;

  /// 优先级
  final int? priority;

  /// 处理人 ID
  final int? assigneeId;
  final String? startTime;
  final String? endTime;

  Map<String, dynamic> toJson() => _$FeedbackQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
    'pageNum': pageNum,
    'pageSize': pageSize,
    if (keywords != null) 'keywords': keywords,
    if (feedbackType != null) 'feedbackType': feedbackType,
    if (status != null) 'status': status,
    if (relatedModule != null) 'relatedModule': relatedModule,
    if (priority != null) 'priority': priority,
    if (assigneeId != null) 'assigneeId': assigneeId,
    if (startTime != null) 'startTime': startTime,
    if (endTime != null) 'endTime': endTime,
  };
}

/// 反馈分页项 VO
@JsonSerializable()
class FeedbackPageVO {
  const FeedbackPageVO({
    required this.id,
    required this.userId,
    required this.username,
    required this.feedbackType,
    required this.title,
    required this.content,
    required this.status,
    required this.priority,
    required this.createTime,
    this.assigneeId,
    this.assigneeName,
    this.relatedModule,
    this.tags,
    this.updateTime,
  });

  factory FeedbackPageVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackPageVOFromJson(json);

  final int id;
  final int userId;
  final String username;

  @JsonKey(fromJson: _feedbackTypeFromJson, toJson: _feedbackTypeToJson)
  final FeedbackType feedbackType;

  final String title;
  final String content;

  @JsonKey(fromJson: _feedbackStatusFromJson, toJson: _feedbackStatusToJson)
  final FeedbackStatus status;

  final int priority;
  final int? assigneeId;
  final String? assigneeName;
  final String? relatedModule;
  final List<String>? tags;
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
    required this.username,
    required this.feedbackType,
    required this.title,
    required this.content,
    required this.status,
    required this.priority,
    required this.createTime,
    this.assigneeId,
    this.assigneeName,
    this.relatedModule,
    this.tags,
    this.updateTime,
    this.contact,
    this.images,
    this.assignedTime,
    this.closeReason,
    this.replies = const [],
  });

  factory FeedbackDetailVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackDetailVOFromJson(json);

  final int id;
  final int userId;
  final String username;

  @JsonKey(fromJson: _feedbackTypeFromJson, toJson: _feedbackTypeToJson)
  final FeedbackType feedbackType;

  final String title;
  final String content;

  @JsonKey(fromJson: _feedbackStatusFromJson, toJson: _feedbackStatusToJson)
  final FeedbackStatus status;

  final int priority;
  final int? assigneeId;
  final String? assigneeName;
  final String? relatedModule;
  final List<String>? tags;
  final String createTime;
  final String? updateTime;

  final String? contact;
  final List<String>? images;
  final String? assignedTime;
  final String? closeReason;
  final List<FeedbackReplyVO> replies;

  Map<String, dynamic> toJson() => _$FeedbackDetailVOToJson(this);
}

/// 反馈回复 VO
@JsonSerializable()
class FeedbackReplyVO {
  const FeedbackReplyVO({
    required this.id,
    required this.feedbackId,
    required this.replierId,
    required this.replierName,
    required this.replierType,
    required this.content,
    required this.createTime,
    this.replyType,
    this.attachments,
  });

  factory FeedbackReplyVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackReplyVOFromJson(json);

  final int id;
  final int feedbackId;
  final int replierId;
  final String replierName;

  @JsonKey(fromJson: _replierTypeFromJson, toJson: _replierTypeToJson)
  final ReplierType replierType;

  final String content;

  @JsonKey(
    fromJson: _feedbackReplyTypeFromJson,
    toJson: _feedbackReplyTypeToJson,
  )
  final FeedbackReplyType? replyType;

  final List<String>? attachments;
  final String createTime;

  Map<String, dynamic> toJson() => _$FeedbackReplyVOToJson(this);
}

/// 补充反馈表单（反馈 ID 走路径）
@JsonSerializable()
class FeedbackSupplementForm {
  const FeedbackSupplementForm({required this.content, this.attachments});

  factory FeedbackSupplementForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackSupplementFormFromJson(json);

  final String content;
  final List<String>? attachments;

  Map<String, dynamic> toJson() => _$FeedbackSupplementFormToJson(this);
}

/// 回复反馈表单（反馈 ID 走路径）
@JsonSerializable()
class FeedbackReplyForm {
  const FeedbackReplyForm({
    required this.content,
    this.replyType,
    this.attachments,
  });

  factory FeedbackReplyForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackReplyFormFromJson(json);

  final String content;

  @JsonKey(
    fromJson: _feedbackReplyTypeFromJson,
    toJson: _feedbackReplyTypeToJson,
  )
  final FeedbackReplyType? replyType;

  final List<String>? attachments;

  Map<String, dynamic> toJson() => _$FeedbackReplyFormToJson(this);
}

/// 分配反馈表单（反馈 ID 走路径）
@JsonSerializable()
class FeedbackAssignForm {
  const FeedbackAssignForm({required this.assigneeId});

  factory FeedbackAssignForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackAssignFormFromJson(json);

  final int assigneeId;

  Map<String, dynamic> toJson() => _$FeedbackAssignFormToJson(this);
}

/// 关闭反馈表单（反馈 ID 走路径）
@JsonSerializable()
class FeedbackCloseForm {
  const FeedbackCloseForm({required this.closeReason});

  factory FeedbackCloseForm.fromJson(Map<String, dynamic> json) =>
      _$FeedbackCloseFormFromJson(json);

  final String closeReason;

  Map<String, dynamic> toJson() => _$FeedbackCloseFormToJson(this);
}

/// 反馈统计 VO
@JsonSerializable()
class FeedbackStatsVO {
  const FeedbackStatsVO({
    required this.totalFeedback,
    required this.averageResponseTime,
    required this.averageCloseTime,
    this.typeDistribution,
    this.moduleDistribution,
    this.statusDistribution,
    this.topKeywords,
  });

  factory FeedbackStatsVO.fromJson(Map<String, dynamic> json) =>
      _$FeedbackStatsVOFromJson(json);

  final int totalFeedback;

  /// 类型分布（类型 → 数量）
  final Map<String, int>? typeDistribution;

  /// 模块分布
  final List<FeedbackModuleCount>? moduleDistribution;

  /// 状态分布（状态 → 数量）
  final Map<String, int>? statusDistribution;

  /// 平均响应时长
  final double averageResponseTime;

  /// 平均关闭时长
  final double averageCloseTime;

  /// 高频关键词
  final List<FeedbackKeywordCount>? topKeywords;

  Map<String, dynamic> toJson() => _$FeedbackStatsVOToJson(this);
}

/// 反馈模块分布项
@JsonSerializable()
class FeedbackModuleCount {
  const FeedbackModuleCount({required this.module, required this.count});

  factory FeedbackModuleCount.fromJson(Map<String, dynamic> json) =>
      _$FeedbackModuleCountFromJson(json);

  final String module;
  final int count;

  Map<String, dynamic> toJson() => _$FeedbackModuleCountToJson(this);
}

/// 反馈关键词统计项
@JsonSerializable()
class FeedbackKeywordCount {
  const FeedbackKeywordCount({required this.keyword, required this.count});

  factory FeedbackKeywordCount.fromJson(Map<String, dynamic> json) =>
      _$FeedbackKeywordCountFromJson(json);

  final String keyword;
  final int count;

  Map<String, dynamic> toJson() => _$FeedbackKeywordCountToJson(this);
}
