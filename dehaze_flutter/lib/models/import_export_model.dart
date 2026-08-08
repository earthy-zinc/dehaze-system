import 'package:json_annotation/json_annotation.dart';

part 'import_export_model.g.dart';

/// 导入记录视图对象
///
/// 对应后端导入任务记录。
@JsonSerializable()
class ImportRecordVO {
  const ImportRecordVO({
    required this.id,
    required this.type,
    this.fileName,
    this.fileSize,
    required this.status,
    this.statusName,
    this.total,
    this.success,
    this.failed,
    this.errorMessage,
    this.createTime,
    this.completeTime,
  });

  factory ImportRecordVO.fromJson(Map<String, dynamic> json) =>
      _$ImportRecordVOFromJson(json);

  final int id;
  final String type;
  final String? fileName;
  final int? fileSize;
  final int status;
  final String? statusName;
  final int? total;
  final int? success;
  final int? failed;
  final String? errorMessage;
  final String? createTime;
  final String? completeTime;

  Map<String, dynamic> toJson() => _$ImportRecordVOToJson(this);
}

/// 导入记录查询参数
@JsonSerializable()
class ImportRecordQuery {
  const ImportRecordQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.type,
    this.status,
    this.startDate,
    this.endDate,
  });

  factory ImportRecordQuery.fromJson(Map<String, dynamic> json) =>
      _$ImportRecordQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? type;
  final int? status;
  final String? startDate;
  final String? endDate;

  Map<String, dynamic> toJson() => _$ImportRecordQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (type != null) 'type': type,
        if (status != null) 'status': status,
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      };
}

/// 导出记录视图对象
///
/// 对应后端导出任务记录。
@JsonSerializable()
class ExportRecordVO {
  const ExportRecordVO({
    required this.id,
    required this.type,
    this.fileName,
    this.fileSize,
    required this.status,
    this.statusName,
    this.total,
    this.downloadUrl,
    this.errorMessage,
    this.createTime,
    this.completeTime,
  });

  factory ExportRecordVO.fromJson(Map<String, dynamic> json) =>
      _$ExportRecordVOFromJson(json);

  final int id;
  final String type;
  final String? fileName;
  final int? fileSize;
  final int status;
  final String? statusName;
  final int? total;
  final String? downloadUrl;
  final String? errorMessage;
  final String? createTime;
  final String? completeTime;

  Map<String, dynamic> toJson() => _$ExportRecordVOToJson(this);
}

/// 导出记录查询参数
@JsonSerializable()
class ExportRecordQuery {
  const ExportRecordQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.type,
    this.status,
    this.startDate,
    this.endDate,
  });

  factory ExportRecordQuery.fromJson(Map<String, dynamic> json) =>
      _$ExportRecordQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final String? type;
  final int? status;
  final String? startDate;
  final String? endDate;

  Map<String, dynamic> toJson() => _$ExportRecordQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (type != null) 'type': type,
        if (status != null) 'status': status,
        if (startDate != null) 'startDate': startDate,
        if (endDate != null) 'endDate': endDate,
      };
}

/// 导入模板视图对象
///
/// 对应后端导入模板元数据。
@JsonSerializable()
class ImportTemplateVO {
  const ImportTemplateVO({
    required this.type,
    required this.name,
    this.description,
    this.url,
    required this.columns,
  });

  factory ImportTemplateVO.fromJson(Map<String, dynamic> json) =>
      _$ImportTemplateVOFromJson(json);

  final String type;
  final String name;
  final String? description;
  final String? url;
  final List<String> columns;

  Map<String, dynamic> toJson() => _$ImportTemplateVOToJson(this);
}
