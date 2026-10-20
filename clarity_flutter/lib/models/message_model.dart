import 'package:json_annotation/json_annotation.dart';

part 'message_model.g.dart';

@JsonSerializable()
class MessageVO {
  const MessageVO({
    required this.id,
    required this.type,
    required this.typeLabel,
    required this.title,
    this.summary,
    this.content,
    required this.priority,
    required this.readStatus,
    required this.senderType,
    this.readTime,
    this.jumpUrl,
    this.extra,
    required this.createTime,
  });

  factory MessageVO.fromJson(Map<String, dynamic> json) =>
      _$MessageVOFromJson(json);

  final int id;
  final String type;

  final String typeLabel;

  final String title;
  final String? summary;
  final String? content;
  final int priority;

  final int readStatus;

  final int senderType;

  final String? readTime;

  final String? jumpUrl;

  final Map<String, dynamic>? extra;

  final String createTime;

  Map<String, dynamic> toJson() => _$MessageVOToJson(this);

  bool get isRead => readStatus == 1;
}

@JsonSerializable()
class UnreadCountVO {
  const UnreadCountVO({required this.count});

  factory UnreadCountVO.fromJson(Map<String, dynamic> json) =>
      _$UnreadCountVOFromJson(json);

  final int count;

  Map<String, dynamic> toJson() => _$UnreadCountVOToJson(this);
}

@JsonSerializable()
class ReadAllResult {
  const ReadAllResult({required this.affectedCount});

  factory ReadAllResult.fromJson(Map<String, dynamic> json) =>
      _$ReadAllResultFromJson(json);

  final int affectedCount;

  Map<String, dynamic> toJson() => _$ReadAllResultToJson(this);
}
