import 'package:dio/dio.dart';

/// 字段校验错误详情
class ApiFieldError {
  const ApiFieldError({
    this.field,
    this.message,
    this.code,
  });

  final String? field;
  final String? message;
  final String? code;
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

/// 从异常对象中提取友好的错误信息。
///
/// 统一错误展示逻辑：
/// - 拦截器层已将业务/网络错误转换为 `DioException(error: ApiException)`
/// - 此函数负责从任意异常中提取 message，避免各页面重复实现
String extractErrorMessage(Object e) {
  if (e is DioException && e.error is ApiException) {
    return (e.error as ApiException).message;
  }
  if (e is ApiException) return e.message;
  return e.toString().replaceFirst('Exception: ', '');
}
