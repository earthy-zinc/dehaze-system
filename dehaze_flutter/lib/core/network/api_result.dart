import 'package:json_annotation/json_annotation.dart';

part 'api_result.g.dart';

/// 统一 API 响应包装
///
/// 对应后端 Result 结构：
/// ```json
/// {
///   "code": "00000",
///   "msg": "一切ok",
///   "data": {},
///   "traceId": "abc123",
///   "timestamp": 1737100800000,
///   "errors": []
/// }
/// ```
@JsonSerializable(genericArgumentFactories: true)
class ApiResult<T> {
  const ApiResult({
    required this.code,
    required this.msg,
    this.data,
    this.traceId,
    this.timestamp,
    this.errors,
  });

  factory ApiResult.fromJson(
    Map<String, dynamic> json,
    T Function(Object? json) fromJsonT,
  ) =>
      _$ApiResultFromJson(json, fromJsonT);

  /// 业务状态码，成功为 "00000"
  @JsonKey(defaultValue: '')
  final String code;

  /// 状态描述信息
  @JsonKey(defaultValue: '')
  final String msg;

  /// 业务数据
  final T? data;

  /// 请求追踪 ID
  final String? traceId;

  /// 响应时间戳（毫秒）
  final int? timestamp;

  /// 错误详情列表（参数校验失败时）
  final List<ApiFieldError>? errors;

  /// 是否成功
  bool get isSuccess => code == '00000';

  /// 是否为认证错误（A02xx）
  bool get isAuthError => code.startsWith('A02');

  /// 是否为权限错误（A03xx）
  bool get isPermissionError => code.startsWith('A03');

  /// 是否为参数错误（A04xx）
  bool get isParamError => code.startsWith('A04');

  Map<String, dynamic> toJson(Object? Function(T value) toJsonT) =>
      _$ApiResultToJson(this, toJsonT);
}

/// 字段校验错误详情
@JsonSerializable()
class ApiFieldError {
  const ApiFieldError({
    this.field,
    this.message,
    this.code,
  });

  factory ApiFieldError.fromJson(Map<String, dynamic> json) =>
      _$ApiFieldErrorFromJson(json);

  final String? field;
  final String? message;
  final String? code;

  Map<String, dynamic> toJson() => _$ApiFieldErrorToJson(this);
}

/// API 业务异常
///
/// 当后端返回 code != "00000" 时抛出
class ApiException implements Exception {
  const ApiException({
    required this.code,
    required this.message,
    this.errors,
  });

  /// 业务状态码
  final String code;

  /// 错误消息
  final String message;

  /// 字段错误列表
  final List<ApiFieldError>? errors;

  /// 是否为认证错误（A02xx）
  bool get isAuthError => code.startsWith('A02');

  /// 是否为权限错误（A03xx）
  bool get isPermissionError => code.startsWith('A03');

  /// 是否为参数错误（A04xx）
  bool get isParamError => code.startsWith('A04');

  @override
  String toString() => 'ApiException($code): $message';
}
