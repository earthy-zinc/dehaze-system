import 'package:json_annotation/json_annotation.dart';

part 'file_model.g.dart';

/// 文件元数据
@JsonSerializable()
class FileModel {
  const FileModel({
    required this.fileId,
    required this.fileName,
    required this.fileUrl,
    required this.fileSize,
    required this.fileType,
    this.md5,
    this.objectName,
    this.createTime,
  });

  factory FileModel.fromJson(Map<String, dynamic> json) =>
      _$FileModelFromJson(json);

  /// 文件 ID
  @JsonKey(name: 'fileId')
  final String fileId;

  /// 文件名
  @JsonKey(name: 'fileName')
  final String fileName;

  /// 文件访问 URL
  @JsonKey(name: 'fileUrl')
  final String fileUrl;

  /// 文件大小（字节）
  @JsonKey(name: 'fileSize')
  final int fileSize;

  /// 文件类型/MIME 类型
  @JsonKey(name: 'fileType')
  final String fileType;

  /// MD5 值
  final String? md5;

  /// 对象存储路径
  @JsonKey(name: 'objectName')
  final String? objectName;

  /// 创建时间
  @JsonKey(name: 'createTime')
  final String? createTime;

  Map<String, dynamic> toJson() => _$FileModelToJson(this);
}

/// 文件上传响应
@JsonSerializable()
class FileUploadResponse {
  const FileUploadResponse({
    required this.fileId,
    required this.fileUrl,
    required this.fileName,
    required this.fileSize,
    this.md5,
  });

  factory FileUploadResponse.fromJson(Map<String, dynamic> json) =>
      _$FileUploadResponseFromJson(json);

  @JsonKey(name: 'fileId')
  final String fileId;

  @JsonKey(name: 'fileUrl')
  final String fileUrl;

  @JsonKey(name: 'fileName')
  final String fileName;

  @JsonKey(name: 'fileSize')
  final int fileSize;

  final String? md5;

  Map<String, dynamic> toJson() => _$FileUploadResponseToJson(this);
}
