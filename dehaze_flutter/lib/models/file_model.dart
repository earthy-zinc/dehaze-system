import 'package:json_annotation/json_annotation.dart';

part 'file_model.g.dart';

/// 文件信息
///
/// 对齐 JS SDK FileInfo 字段。
@JsonSerializable()
class FileInfo {
  const FileInfo({
    required this.id,
    required this.url,
    required this.name,
    this.type,
    this.objectName,
    this.size,
    this.sizeBytes,
    this.storage,
    this.md5,
    this.createTime,
  });

  factory FileInfo.fromJson(Map<String, dynamic> json) =>
      _$FileInfoFromJson(json);

  /// 文件 ID
  final int id;

  /// 文件访问 URL
  final String url;

  /// 文件名
  final String name;

  /// 文件类型（扩展名）
  final String? type;

  /// 文件存储对象名（与环境无关）
  final String? objectName;

  /// 文件大小（格式化显示，如 "2.44MB"）
  final String? size;

  /// 文件大小（原始字节数）
  final int? sizeBytes;

  /// 存储后端标识（minio / local / nginx-static）
  final String? storage;

  /// 文件 MD5 值
  final String? md5;

  /// 创建时间
  final String? createTime;

  Map<String, dynamic> toJson() => _$FileInfoToJson(this);
}

/// 文件查询参数
///
/// 对齐 JS SDK FileQuery。
class FileQuery {
  const FileQuery({
    this.pageNum,
    this.pageSize,
    this.keywords,
  });

  /// 页码
  final int? pageNum;

  /// 每页条数
  final int? pageSize;

  /// 关键字搜索
  final String? keywords;

  Map<String, dynamic> toJson() => {
        if (pageNum != null) 'pageNum': pageNum,
        if (pageSize != null) 'pageSize': pageSize,
        if (keywords != null) 'keywords': keywords,
      };
}
