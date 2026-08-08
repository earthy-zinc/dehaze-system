// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'feedback_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RatingCreateForm _$RatingCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('RatingCreateForm', json, ($checkedConvert) {
      final val = RatingCreateForm(
        predictionLogId: $checkedConvert(
          'predictionLogId',
          (v) => (v as num).toInt(),
        ),
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$RatingCreateFormToJson(RatingCreateForm instance) =>
    <String, dynamic>{
      'predictionLogId': instance.predictionLogId,
      'algorithmId': instance.algorithmId,
      'rating': instance.rating,
      if (instance.comment case final value?) 'comment': value,
    };

RatingQuery _$RatingQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'RatingQuery',
  json,
  ($checkedConvert) {
    final val = RatingQuery(
      algorithmId: $checkedConvert('algorithmId', (v) => (v as num?)?.toInt()),
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
      if (instance.algorithmId case final value?) 'algorithmId': value,
      if (instance.userId case final value?) 'userId': value,
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.minRating case final value?) 'minRating': value,
      if (instance.maxRating case final value?) 'maxRating': value,
      if (instance.sortBy case final value?) 'sortBy': value,
      if (instance.sortOrder case final value?) 'sortOrder': value,
    };

MyRatingVO _$MyRatingVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('MyRatingVO', json, ($checkedConvert) {
      final val = MyRatingVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        predictionLogId: $checkedConvert(
          'predictionLogId',
          (v) => (v as num).toInt(),
        ),
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        rating: $checkedConvert('rating', (v) => (v as num).toInt()),
        comment: $checkedConvert('comment', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$MyRatingVOToJson(MyRatingVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'predictionLogId': instance.predictionLogId,
      'algorithmId': instance.algorithmId,
      'algorithmName': instance.algorithmName,
      'rating': instance.rating,
      if (instance.comment case final value?) 'comment': value,
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
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        predictionLogId: $checkedConvert(
          'predictionLogId',
          (v) => (v as num).toInt(),
        ),
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
      if (instance.username case final value?) 'username': value,
      if (instance.nickname case final value?) 'nickname': value,
      if (instance.avatar case final value?) 'avatar': value,
      'algorithmId': instance.algorithmId,
      'algorithmName': instance.algorithmName,
      'predictionLogId': instance.predictionLogId,
      'rating': instance.rating,
      if (instance.comment case final value?) 'comment': value,
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
        algorithmId: $checkedConvert('algorithmId', (v) => (v as num).toInt()),
        algorithmName: $checkedConvert('algorithmName', (v) => v as String),
        predictionLogId: $checkedConvert(
          'predictionLogId',
          (v) => (v as num).toInt(),
        ),
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
      if (instance.username case final value?) 'username': value,
      if (instance.nickname case final value?) 'nickname': value,
      if (instance.avatar case final value?) 'avatar': value,
      'algorithmId': instance.algorithmId,
      'algorithmName': instance.algorithmName,
      'predictionLogId': instance.predictionLogId,
      'rating': instance.rating,
      if (instance.comment case final value?) 'comment': value,
      'createTime': instance.createTime,
      if (instance.predictionUrl case final value?) 'predictionUrl': value,
      if (instance.resultUrl case final value?) 'resultUrl': value,
      if (instance.feedback case final value?) 'feedback': value,
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
        type: $checkedConvert('type', (v) => v as String),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        contact: $checkedConvert('contact', (v) => v as String?),
        images: $checkedConvert(
          'images',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackCreateFormToJson(FeedbackCreateForm instance) =>
    <String, dynamic>{
      'type': instance.type,
      'title': instance.title,
      'content': instance.content,
      if (instance.contact case final value?) 'contact': value,
      if (instance.images case final value?) 'images': value,
    };

FeedbackQuery _$FeedbackQueryFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackQuery', json, ($checkedConvert) {
      final val = FeedbackQuery(
        type: $checkedConvert('type', (v) => v as String?),
        status: $checkedConvert('status', (v) => v as String?),
        keyword: $checkedConvert('keyword', (v) => v as String?),
        pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
        pageSize: $checkedConvert(
          'pageSize',
          (v) => (v as num?)?.toInt() ?? 10,
        ),
        userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
        startDate: $checkedConvert('startDate', (v) => v as String?),
        endDate: $checkedConvert('endDate', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackQueryToJson(FeedbackQuery instance) =>
    <String, dynamic>{
      if (instance.type case final value?) 'type': value,
      if (instance.status case final value?) 'status': value,
      if (instance.keyword case final value?) 'keyword': value,
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.userId case final value?) 'userId': value,
      if (instance.startDate case final value?) 'startDate': value,
      if (instance.endDate case final value?) 'endDate': value,
    };

FeedbackPageVO _$FeedbackPageVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackPageVO', json, ($checkedConvert) {
      final val = FeedbackPageVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String?),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        type: $checkedConvert('type', (v) => v as String),
        typeName: $checkedConvert('typeName', (v) => v as String?),
        title: $checkedConvert('title', (v) => v as String),
        status: $checkedConvert('status', (v) => v as String),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackPageVOToJson(FeedbackPageVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      if (instance.username case final value?) 'username': value,
      if (instance.nickname case final value?) 'nickname': value,
      'type': instance.type,
      if (instance.typeName case final value?) 'typeName': value,
      'title': instance.title,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      'createTime': instance.createTime,
      if (instance.updateTime case final value?) 'updateTime': value,
    };

FeedbackDetailVO _$FeedbackDetailVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackDetailVO', json, ($checkedConvert) {
      final val = FeedbackDetailVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        userId: $checkedConvert('userId', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String?),
        nickname: $checkedConvert('nickname', (v) => v as String?),
        avatar: $checkedConvert('avatar', (v) => v as String?),
        type: $checkedConvert('type', (v) => v as String),
        typeName: $checkedConvert('typeName', (v) => v as String?),
        title: $checkedConvert('title', (v) => v as String),
        content: $checkedConvert('content', (v) => v as String),
        contact: $checkedConvert('contact', (v) => v as String?),
        images: $checkedConvert(
          'images',
          (v) =>
              (v as List<dynamic>?)?.map((e) => e as String).toList() ??
              const [],
        ),
        status: $checkedConvert('status', (v) => v as String),
        statusName: $checkedConvert('statusName', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
        replies: $checkedConvert(
          'replies',
          (v) => (v as List<dynamic>?)
              ?.map((e) => FeedbackReplyVO.fromJson(e as Map<String, dynamic>))
              .toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackDetailVOToJson(FeedbackDetailVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'userId': instance.userId,
      if (instance.username case final value?) 'username': value,
      if (instance.nickname case final value?) 'nickname': value,
      if (instance.avatar case final value?) 'avatar': value,
      'type': instance.type,
      if (instance.typeName case final value?) 'typeName': value,
      'title': instance.title,
      'content': instance.content,
      if (instance.contact case final value?) 'contact': value,
      'images': instance.images,
      'status': instance.status,
      if (instance.statusName case final value?) 'statusName': value,
      'createTime': instance.createTime,
      if (instance.updateTime case final value?) 'updateTime': value,
      if (instance.replies?.map((e) => e.toJson()).toList() case final value?)
        'replies': value,
    };

FeedbackReplyVO _$FeedbackReplyVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackReplyVO', json, ($checkedConvert) {
      final val = FeedbackReplyVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
        content: $checkedConvert('content', (v) => v as String),
        replyType: $checkedConvert('replyType', (v) => v as String?),
        replierType: $checkedConvert('replierType', (v) => v as String?),
        replierId: $checkedConvert('replierId', (v) => (v as num?)?.toInt()),
        replierName: $checkedConvert('replierName', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackReplyVOToJson(FeedbackReplyVO instance) =>
    <String, dynamic>{
      'id': instance.id,
      'feedbackId': instance.feedbackId,
      'content': instance.content,
      if (instance.replyType case final value?) 'replyType': value,
      if (instance.replierType case final value?) 'replierType': value,
      if (instance.replierId case final value?) 'replierId': value,
      if (instance.replierName case final value?) 'replierName': value,
      'createTime': instance.createTime,
    };

FeedbackSupplementForm _$FeedbackSupplementFormFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('FeedbackSupplementForm', json, ($checkedConvert) {
  final val = FeedbackSupplementForm(
    feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
    content: $checkedConvert('content', (v) => v as String),
    images: $checkedConvert(
      'images',
      (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
    ),
  );
  return val;
});

Map<String, dynamic> _$FeedbackSupplementFormToJson(
  FeedbackSupplementForm instance,
) => <String, dynamic>{
  'feedbackId': instance.feedbackId,
  'content': instance.content,
  if (instance.images case final value?) 'images': value,
};

FeedbackReplyForm _$FeedbackReplyFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackReplyForm', json, ($checkedConvert) {
      final val = FeedbackReplyForm(
        feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
        content: $checkedConvert('content', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackReplyFormToJson(FeedbackReplyForm instance) =>
    <String, dynamic>{
      'feedbackId': instance.feedbackId,
      'content': instance.content,
    };

FeedbackAssignForm _$FeedbackAssignFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackAssignForm', json, ($checkedConvert) {
      final val = FeedbackAssignForm(
        feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
        assigneeId: $checkedConvert('assigneeId', (v) => (v as num).toInt()),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackAssignFormToJson(FeedbackAssignForm instance) =>
    <String, dynamic>{
      'feedbackId': instance.feedbackId,
      'assigneeId': instance.assigneeId,
    };

FeedbackCloseForm _$FeedbackCloseFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackCloseForm', json, ($checkedConvert) {
      final val = FeedbackCloseForm(
        feedbackId: $checkedConvert('feedbackId', (v) => (v as num).toInt()),
        reason: $checkedConvert('reason', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackCloseFormToJson(FeedbackCloseForm instance) =>
    <String, dynamic>{
      'feedbackId': instance.feedbackId,
      'reason': instance.reason,
    };

FeedbackStatsVO _$FeedbackStatsVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('FeedbackStatsVO', json, ($checkedConvert) {
      final val = FeedbackStatsVO(
        totalFeedbacks: $checkedConvert(
          'totalFeedbacks',
          (v) => (v as num).toInt(),
        ),
        pendingFeedbacks: $checkedConvert(
          'pendingFeedbacks',
          (v) => (v as num).toInt(),
        ),
        resolvedFeedbacks: $checkedConvert(
          'resolvedFeedbacks',
          (v) => (v as num).toInt(),
        ),
        closedFeedbacks: $checkedConvert(
          'closedFeedbacks',
          (v) => (v as num).toInt(),
        ),
        typeDistribution: $checkedConvert(
          'typeDistribution',
          (v) => Map<String, int>.from(v as Map),
        ),
        statusDistribution: $checkedConvert(
          'statusDistribution',
          (v) => Map<String, int>.from(v as Map),
        ),
      );
      return val;
    });

Map<String, dynamic> _$FeedbackStatsVOToJson(FeedbackStatsVO instance) =>
    <String, dynamic>{
      'totalFeedbacks': instance.totalFeedbacks,
      'pendingFeedbacks': instance.pendingFeedbacks,
      'resolvedFeedbacks': instance.resolvedFeedbacks,
      'closedFeedbacks': instance.closedFeedbacks,
      'typeDistribution': instance.typeDistribution,
      'statusDistribution': instance.statusDistribution,
    };
