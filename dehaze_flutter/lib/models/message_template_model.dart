import 'package:json_annotation/json_annotation.dart';

part 'message_template_model.g.dart';

// ==================== 消息模板 VO ====================

@JsonSerializable()
class MessageTemplateVO {
  const MessageTemplateVO({
    required this.id,
    required this.code,
    required this.name,
    required this.title,
    required this.content,
    required this.type,
    this.typeName,
    this.variables,
    required this.status,
    this.statusName,
    this.description,
    required this.createTime,
    this.updateTime,
  });

  factory MessageTemplateVO.fromJson(Map<String, dynamic> json) =>
      _$MessageTemplateVOFromJson(json);

  final int id;
  final String code;
  final String name;
  final String title;
  final String content;
  final String type;
  final String? typeName;
  final List<String>? variables;
  final String status;
  final String? statusName;
  final String? description;
  final String createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$MessageTemplateVOToJson(this);
}

// ==================== 消息模板查询参数 ====================

@JsonSerializable()
class MessageTemplateQuery {
  const MessageTemplateQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.code,
    this.name,
    this.type,
    this.status,
  });

  factory MessageTemplateQuery.fromJson(Map<String, dynamic> json) =>
      _$MessageTemplateQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? code;
  final String? name;
  final String? type;
  final String? status;

  Map<String, dynamic> toJson() => _$MessageTemplateQueryToJson(this);

  /// 转为 queryParameters 格式（仅非 null 字段）
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (code != null) 'code': code,
        if (name != null) 'name': name,
        if (type != null) 'type': type,
        if (status != null) 'status': status,
      };
}

// ==================== 消息模板表单 ====================

@JsonSerializable()
class MessageTemplateForm {
  const MessageTemplateForm({
    this.id,
    required this.code,
    required this.name,
    required this.title,
    required this.content,
    required this.type,
    this.variables,
    required this.status,
    this.description,
  });

  factory MessageTemplateForm.fromJson(Map<String, dynamic> json) =>
      _$MessageTemplateFormFromJson(json);

  final int? id;
  final String code;
  final String name;
  final String title;
  final String content;
  final String type;
  final List<String>? variables;
  final String status;
  final String? description;

  Map<String, dynamic> toJson() => _$MessageTemplateFormToJson(this);
}
