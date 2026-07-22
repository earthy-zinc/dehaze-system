import 'package:json_annotation/json_annotation.dart';

part 'file_model.g.dart';

/// 文件上传响应
///
/// 对应后端 SysFile：POST /files 返回。
@JsonSerializable()
class FileUploadResponse {
  const FileUploadResponse({
    required this.id,
    required this.url,
    required this.name,
    this.type,
    this.objectName,
    this.size,
    this.path,
    this.md5,
  });

  factory FileUploadResponse.fromJson(Map<String, dynamic> json) =>
      _$FileUploadResponseFromJson(json);

  /// 文件 ID
  final int id;

  /// 文件访问 URL
  final String url;

  /// 文件名
  final String name;

  /// 文件类型/MIME 类型
  final String? type;

  /// 对象存储路径
  @JsonKey(name: 'objectName')
  final String? objectName;

  /// 文件大小（格式化显示，如 "2.44MB"）
  final String? size;

  /// 文件路径
  final String? path;

  /// MD5 值
  final String? md5;

  Map<String, dynamic> toJson() => _$FileUploadResponseToJson(this);
}
