import 'dart:convert';

import 'package:json_annotation/json_annotation.dart';

part 'prediction_model.g.dart';

/// 预测请求
///
/// 对应后端 PredictionForm：
/// algorithmId（Long，必填）、fileId（Long）、imageUrl（String）、params（String，JSON）
@JsonSerializable()
class PredictionRequest {
  const PredictionRequest({
    required this.algorithmId,
    required this.fileId,
    this.params,
  });

  factory PredictionRequest.fromJson(Map<String, dynamic> json) =>
      _$PredictionRequestFromJson(json);

  /// 算法 ID
  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  /// 原始图片文件 ID
  @JsonKey(name: 'fileId')
  final int fileId;

  /// 算法参数
  ///
  /// 后端 `params` 字段为 String 类型，需序列化为 JSON 字符串传输。
  @JsonKey(toJson: _paramsToJson, fromJson: _paramsFromJson)
  final Map<String, dynamic>? params;

  static String? _paramsToJson(Map<String, dynamic>? params) =>
      params == null ? null : jsonEncode(params);

  static Map<String, dynamic>? _paramsFromJson(String? params) =>
      params == null || params.isEmpty
          ? null
          : jsonDecode(params) as Map<String, dynamic>;

  Map<String, dynamic> toJson() => _$PredictionRequestToJson(this);
}

/// 预测响应
///
/// 对应后端 PredictionResultVO：预测为同步接口，直接返回结果。
@JsonSerializable()
class PredictionResponse {
  const PredictionResponse({
    required this.logId,
    this.resultUrl,
    this.resultThumbnailUrl,
    this.time,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) =>
      _$PredictionResponseFromJson(json);

  /// 预测日志 ID
  @JsonKey(name: 'logId')
  final int logId;

  /// 结果图片 URL
  @JsonKey(name: 'resultUrl')
  final String? resultUrl;

  /// 结果缩略图 URL
  @JsonKey(name: 'resultThumbnailUrl')
  final String? resultThumbnailUrl;

  /// 处理耗时（毫秒）
  final int? time;

  Map<String, dynamic> toJson() => _$PredictionResponseToJson(this);

  /// 是否拿到处理结果
  bool get hasResult => resultUrl != null && resultUrl!.isNotEmpty;
}

/// 预测日志
///
/// 对应后端 PredLogVO。
@JsonSerializable()
class PredictionLog {
  const PredictionLog({
    required this.id,
    required this.algorithmName,
    required this.createTime,
    this.algorithmId,
    this.originUrl,
    this.predUrl,
    this.time,
  });

  factory PredictionLog.fromJson(Map<String, dynamic> json) =>
      _$PredictionLogFromJson(json);

  final int id;

  @JsonKey(name: 'algorithmId')
  final int? algorithmId;

  @JsonKey(name: 'algorithmName')
  final String algorithmName;

  /// 原始图片 URL
  @JsonKey(name: 'originUrl')
  final String? originUrl;

  /// 预测结果图片 URL（失败的记录可能为空）
  @JsonKey(name: 'predUrl')
  final String? predUrl;

  /// 处理耗时（毫秒）
  final int? time;

  /// 创建时间
  @JsonKey(name: 'createTime')
  final String createTime;

  Map<String, dynamic> toJson() => _$PredictionLogToJson(this);
}
