// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'feedback_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RatingCreateForm _$RatingCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RatingCreateForm', json, ($checkedConvert) {
      final val = RatingCreateForm(
        predLogId: $checkedConvert('predLogId', (v) => (v as num).toInt()),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RatingCreateFormToJson(RatingCreateForm instance) =>
    <String, dynamic>{
      'predLogId': instance.predLogId,
      'rating': instance.rating,
      'comment': ?instance.comment,
    };

RatingQuery _$RatingQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'RatingQuery',
  json,
  ($checkedConvert) {
    final val = RatingQuery(
      modelId: $checkedConvert('modelId', (v) => v as String?),
      userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      minRating: $checkedConvert('minRating', (v) => (v as num?)?.toInt()),
      maxRating: $checkedConvert('maxRating', (v) => (v as num?)?.toInt()),
      sortBy: $checkedConvert('sortBy', (v) => v as String?),
      sortOrder: $checkedConvert('sortOrder', (v) => v as String?),
    );
    return val;
  },
);

Map<String, dynamic> _$RatingQueryToJson(RatingQuery instance) =>
    <String, dynamic>{
      'modelId': ?instance.modelId,
      'userId': ?instance.userId,
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'minRating': ?instance.minRating,
      'maxRating': ?instance.maxRating,
      'sortBy': ?instance.sortBy,
      'sortOrder': ?instance.sortOrder,
    };

MyRatingVO _$MyRatingVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MyRatingVO', json, ($checkedConvert) {
      final val = MyRatingVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        predLogId: $checkedConvert('predLogId', (v) => (v as num).toInt()),
        modelName: $checkedConvert('modelName', (v) => v as String),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$MyRatingVOToJson(MyRatingVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'predLogId': instance.predLogId,
      'modelName': instance.modelName,
      'rating': instance.rating,
      'comment': ?instance.comment,
      'createTime': instance.createTime,
    };

RatingPageVO _$RatingPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RatingPageVO', json, ($checkedConvert) {
      final val = RatingPageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String?),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        modelName: $checkedConvert('modelName', (v) => v as String),
        predLogId: $checkedConvert('predLogId', (v) => (v as num).toInt()),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$RatingPageVOToJson(RatingPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      'username': ?instance.username,
      'nickname': ?instance.nickname,
      'avatar': ?instance.avatar,
      'modelName': instance.modelName,
      'predLogId': instance.predLogId,
      'rating': instance.rating,
      'comment': ?instance.comment,
      'createTime': instance.createTime,
    };

RatingDetailVO _$RatingDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RatingDetailVO', json, ($checkedConvert) {
      final val = RatingDetailVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String?),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        modelId: $checkedConvert('modelId', (v) => v as String),
        modelName: $checkedConvert('modelName', (v) => v as String),
        predLogId: $checkedConvert('predLogId', (v) => (v as num).toInt()),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
        predictionUrl: $checkedConvert('predictionUrl', (v) => v as String?),
        resultUrl: $checkedConvert('resultUrl', (v) => v as String?),
        feedback: $checkedConvert('feedback', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RatingDetailVOToJson(RatingDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      'username': ?instance.username,
      'nickname': ?instance.nickname,
      'avatar': ?instance.avatar,
      'modelId': instance.modelId,
      'modelName': instance.modelName,
      'predLogId': instance.predLogId,
      'rating': instance.rating,
      'comment': ?instance.comment,
      'createTime': instance.createTime,
      'predictionUrl': ?instance.predictionUrl,
      'resultUrl': ?instance.resultUrl,
      'feedback': ?instance.feedback,
    };

RatingStatsVO _$RatingStatsVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RatingStatsVO', json, ($checkedConvert) {
      final val = RatingStatsVO(
        totalRatings: $checkedConvert(
          'totalRatings',
          (v) => (v as num).toInt(),
        ),
        averageRating: $checkedConvert(
          'averageRating',
          (v) => (v as num).toDouble(),
        ),
        fiveStar: $checkedConvert('fiveStar', (v) => (v as num).toInt()),
        fourStar: $checkedConvert('fourStar', (v) => (v as num).toInt()),
        threeStar: $checkedConvert('threeStar', (v) => (v as num).toInt()),
        twoStar: $checkedConvert('twoStar', (v) => (v as num).toInt()),
        oneStar: $checkedConvert('oneStar', (v) => (v as num).toInt()),
        distribution: $checkedConvert(
          'distribution',
          (v) => Map<String, int>.from(v as Map),
        ),
      );
      return val;
    });

Map<String, dynamic> _$RatingStatsVOToJson(RatingStatsVO instance) =>
    <String, dynamic>{
      'totalRatings': instance.totalRatings,
      'averageRating': instance.averageRating,
      'fiveStar': instance.fiveStar,
      'fourStar': instance.fourStar,
      'threeStar': instance.threeStar,
      'twoStar': instance.twoStar,
      'oneStar': instance.oneStar,
      'distribution': instance.distribution,
    };

FeedbackCreateForm _$FeedbackCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackCreateForm', json, ($checkedConvert) {
      final val = FeedbackCreateForm(
        feedbackType: $checkedConvert(
          'feedbackType',
          (v) => _feedbackTypeFromJson(v),
        ),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        contact: $checkedConvert('contact', (v) => v as String?),
        images: $checkedConvert(
          'images',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        relatedModule: $checkedConvert('relatedModule', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackCreateFormToJson(FeedbackCreateForm instance) =>
    <String, dynamic>{
      'feedbackType': _feedbackTypeToJson(instance.feedbackType),
      'title': instance.title,
      'content': instance.content,
      'contact': ?instance.contact,
      'images': ?instance.images,
      'relatedModule': ?instance.relatedModule,
    };

FeedbackQuery _$FeedbackQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackQuery', json, ($checkedConvert) {
      final val = FeedbackQuery(
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        keywords: $checkedConvert('keywords', (v) => v as String?),
        feedbackType: $checkedConvert('feedbackType', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
        relatedModule: $checkedConvert('relatedModule', (v) => v as String?),
        priority: $checkedConvert('priority', (v) => (v as num?)?.toInt()),
        assigneeId: $checkedConvert('assigneeId', (v) => (v as num?)?.toInt()),
        startTime: $checkedConvert('startTime', (v) => v as String?),
        endTime: $checkedConvert('endTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackQueryToJson(FeedbackQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      'keywords': ?instance.keywords,
      'feedbackType': ?instance.feedbackType,
      'status': ?instance.status,
      'relatedModule': ?instance.relatedModule,
      'priority': ?instance.priority,
      'assigneeId': ?instance.assigneeId,
      'startTime': ?instance.startTime,
      'endTime': ?instance.endTime,
    };

FeedbackPageVO _$FeedbackPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackPageVO', json, ($checkedConvert) {
      final val = FeedbackPageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        feedbackType: $checkedConvert(
          'feedbackType',
          (v) => _feedbackTypeFromJson(v),
        ),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        status: $checkedConvert('status', (v) => _feedbackStatusFromJson(v)),
        priority: $checkedConvert('priority', (v) => (v as num).toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String),
        assigneeId: $checkedConvert('assigneeId', (v) => (v as num?)?.toInt()),
        assigneeName: $checkedConvert('assigneeName', (v) => v as String?),
        relatedModule: $checkedConvert('relatedModule', (v) => v as String?),
        tags: $checkedConvert(
          'tags',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackPageVOToJson(FeedbackPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      'username': instance.username,
      'feedbackType': _feedbackTypeToJson(instance.feedbackType),
      'title': instance.title,
      'content': instance.content,
      'status': _feedbackStatusToJson(instance.status),
      'priority': instance.priority,
      'assigneeId': ?instance.assigneeId,
      'assigneeName': ?instance.assigneeName,
      'relatedModule': ?instance.relatedModule,
      'tags': ?instance.tags,
      'createTime': instance.createTime,
      'updateTime': ?instance.updateTime,
    };

FeedbackDetailVO _$FeedbackDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackDetailVO', json, ($checkedConvert) {
      final val = FeedbackDetailVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        feedbackType: $checkedConvert(
          'feedbackType',
          (v) => _feedbackTypeFromJson(v),
        ),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        status: $checkedConvert('status', (v) => _feedbackStatusFromJson(v)),
        priority: $checkedConvert('priority', (v) => (v as num).toInt()),
        createTime: $checkedConvert('createTime', (v) => v as String),
        assigneeId: $checkedConvert('assigneeId', (v) => (v as num?)?.toInt()),
        assigneeName: $checkedConvert('assigneeName', (v) => v as String?),
        relatedModule: $checkedConvert('relatedModule', (v) => v as String?),
        tags: $checkedConvert(
          'tags',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
        contact: $checkedConvert('contact', (v) => v as String?),
        images: $checkedConvert(
          'images',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        assignedTime: $checkedConvert('assignedTime', (v) => v as String?),
        closeReason: $checkedConvert('closeReason', (v) => v as String?),
        replies: $checkedConvert(
          'replies',
          (v) =>
              (v as List<dynamic>?)
                  ?.map(
                    (e) => FeedbackReplyVO.fromJson(e as Map<String, dynamic>),
                  )
                  .toList() ??
              const [],
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackDetailVOToJson(FeedbackDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      'username': instance.username,
      'feedbackType': _feedbackTypeToJson(instance.feedbackType),
      'title': instance.title,
      'content': instance.content,
      'status': _feedbackStatusToJson(instance.status),
      'priority': instance.priority,
      'assigneeId': ?instance.assigneeId,
      'assigneeName': ?instance.assigneeName,
      'relatedModule': ?instance.relatedModule,
      'tags': ?instance.tags,
      'createTime': instance.createTime,
      'updateTime': ?instance.updateTime,
      'contact': ?instance.contact,
      'images': ?instance.images,
      'assignedTime': ?instance.assignedTime,
      'closeReason': ?instance.closeReason,
      'replies': instance.replies.map((e) => e.toJson()).toList(),
    };

FeedbackReplyVO _$FeedbackReplyVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackReplyVO', json, ($checkedConvert) {
      final val = FeedbackReplyVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
        replierId: $checkedConvert('replierId', (v) => (v as num).toInt()),
        replierName: $checkedConvert('replierName', (v) => v as String),
        replierType: $checkedConvert(
          'replierType',
          (v) => _replierTypeFromJson(v),
        ),
        content: $checkedConvert('content', (v) => v as String),
        createTime: $checkedConvert('createTime', (v) => v as String),
        replyType: $checkedConvert(
          'replyType',
          (v) => _feedbackReplyTypeFromJson(v),
        ),
        attachments: $checkedConvert(
          'attachments',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackReplyVOToJson(FeedbackReplyVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'feedbackId': instance.feedbackId,
      'replierId': instance.replierId,
      'replierName': instance.replierName,
      'replierType': _replierTypeToJson(instance.replierType),
      'content': instance.content,
      'replyType': ?_feedbackReplyTypeToJson(instance.replyType),
      'attachments': ?instance.attachments,
      'createTime': instance.createTime,
    };

FeedbackSupplementForm _$FeedbackSupplementFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('FeedbackSupplementForm', json, ($checkedConvert) {
  final val = FeedbackSupplementForm(
    content: $checkedConvert('content', (v) => v as String),
    attachments: $checkedConvert(
      'attachments',
      (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$FeedbackSupplementFormToJson(
  FeedbackSupplementForm instance,
) => <String, dynamic>{
  'content': instance.content,
  'attachments': ?instance.attachments,
};

FeedbackReplyForm _$FeedbackReplyFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackReplyForm', json, ($checkedConvert) {
      final val = FeedbackReplyForm(
        content: $checkedConvert('content', (v) => v as String),
        replyType: $checkedConvert(
          'replyType',
          (v) => _feedbackReplyTypeFromJson(v),
        ),
        attachments: $checkedConvert(
          'attachments',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackReplyFormToJson(FeedbackReplyForm instance) =>
    <String, dynamic>{
      'content': instance.content,
      'replyType': ?_feedbackReplyTypeToJson(instance.replyType),
      'attachments': ?instance.attachments,
    };

FeedbackAssignForm _$FeedbackAssignFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackAssignForm', json, ($checkedConvert) {
      final val = FeedbackAssignForm(
        assigneeId: $checkedConvert('assigneeId', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackAssignFormToJson(FeedbackAssignForm instance) =>
    <String, dynamic>{'assigneeId': instance.assigneeId};

FeedbackCloseForm _$FeedbackCloseFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackCloseForm', json, ($checkedConvert) {
      final val = FeedbackCloseForm(
        closeReason: $checkedConvert('closeReason', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackCloseFormToJson(FeedbackCloseForm instance) =>
    <String, dynamic>{'closeReason': instance.closeReason};

FeedbackStatsVO _$FeedbackStatsVOFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('FeedbackStatsVO', json, ($checkedConvert) {
  final val = FeedbackStatsVO(
    totalFeedback: $checkedConvert('totalFeedback', (v) => (v as num).toInt()),
    averageResponseTime: $checkedConvert(
      'averageResponseTime',
      (v) => (v as num).toDouble(),
    ),
    averageCloseTime: $checkedConvert(
      'averageCloseTime',
      (v) => (v as num).toDouble(),
    ),
    typeDistribution: $checkedConvert(
      'typeDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    moduleDistribution: $checkedConvert(
      'moduleDistribution',
      (v) => (v as List<dynamic>?)
          ?.map((e) => FeedbackModuleCount.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
    statusDistribution: $checkedConvert(
      'statusDistribution',
      (v) => (v as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, (e as num).toInt()),
      ),
    ),
    topKeywords: $checkedConvert(
      'topKeywords',
      (v) => (v as List<dynamic>?)
          ?.map((e) => FeedbackKeywordCount.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$FeedbackStatsVOToJson(FeedbackStatsVO instance) =>
    <String, dynamic>{
      'totalFeedback': instance.totalFeedback,
      'typeDistribution': ?instance.typeDistribution,
      'moduleDistribution': ?instance.moduleDistribution
          ?.map((e) => e.toJson())
          .toList(),
      'statusDistribution': ?instance.statusDistribution,
      'averageResponseTime': instance.averageResponseTime,
      'averageCloseTime': instance.averageCloseTime,
      'topKeywords': ?instance.topKeywords?.map((e) => e.toJson()).toList(),
    };

FeedbackModuleCount _$FeedbackModuleCountFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackModuleCount', json, ($checkedConvert) {
      final val = FeedbackModuleCount(
        module: $checkedConvert('module', (v) => v as String),
        count: $checkedConvert('count', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackModuleCountToJson(
  FeedbackModuleCount instance,
) => <String, dynamic>{'module': instance.module, 'count': instance.count};

FeedbackKeywordCount _$FeedbackKeywordCountFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('FeedbackKeywordCount', json, ($checkedConvert) {
  final val = FeedbackKeywordCount(
    keyword: $checkedConvert('keyword', (v) => v as String),
    count: $checkedConvert('count', (v) => (v as num).toInt()),
  );
  return val;
});

Map<String, dynamic> _$FeedbackKeywordCountToJson(
  FeedbackKeywordCount instance,
) => <String, dynamic>{'keyword': instance.keyword, 'count': instance.count};
