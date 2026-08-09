import 'package:dio/dio.dart';

import '../api_result.dart';

/// 错误拦截器
///
/// 统一处理各类网络/业务错误，转换为友好的异常信息
class ErrorInterceptor extends Interceptor {
  ErrorInterceptor({
    this.onAuthError,
    this.authErrorDebounce = const Duration(seconds: 2),
  });

  /// 认证失败回调（401 或 token 过期）
  final void Function()? onAuthError;

  /// 认证错误防抖窗口，避免并发请求同时 401 时多次触发回调
  final Duration authErrorDebounce;
  DateTime? _lastAuthErrorTime;

  void _triggerAuthError() {
    final now = DateTime.now();
    final last = _lastAuthErrorTime;
    if (last != null && now.difference(last) < authErrorDebounce) {
      return;
    }
    _lastAuthErrorTime = now;
    onAuthError?.call();
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    final error = _transformError(err);
    handler.next(error);
  }

  /// 转换错误为更友好的形式
  DioException _transformError(DioException err) {
    // 已是 ApiException 的直接返回
    if (err.error is ApiException) {
      final apiError = err.error as ApiException;
      // 认证/权限错误触发回调
      if (apiError.isAuthError) {
        _triggerAuthError();
      }
      return err;
    }

    // 根据 Dio 错误类型转换
    switch (err.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        return DioException(
          requestOptions: err.requestOptions,
          type: err.type,
          error: const ApiException(
            code: 'B0100',
            message: '网络请求超时，请检查网络连接后重试',
          ),
        );

      case DioExceptionType.connectionError:
        return DioException(
          requestOptions: err.requestOptions,
          type: err.type,
          error: const ApiException(
            code: 'B0001',
            message: '无法连接到服务器，请检查网络或稍后重试',
          ),
        );

      case DioExceptionType.badResponse:
        final statusCode = err.response?.statusCode;
        if (statusCode == 401) {
          _triggerAuthError();
          return DioException(
            requestOptions: err.requestOptions,
            response: err.response,
            type: err.type,
            error: const ApiException(
              code: 'A0230',
              message: '登录已过期，请重新登录',
            ),
          );
        }
        if (statusCode == 403) {
          return DioException(
            requestOptions: err.requestOptions,
            response: err.response,
            type: err.type,
            error: const ApiException(
              code: 'A0300',
              message: '您没有权限执行此操作',
            ),
          );
        }
        if (statusCode == 404) {
          return DioException(
            requestOptions: err.requestOptions,
            response: err.response,
            type: err.type,
            error: const ApiException(
              code: 'A0401',
              message: '请求的资源不存在',
            ),
          );
        }
        if (statusCode != null && statusCode >= 500) {
          return DioException(
            requestOptions: err.requestOptions,
            response: err.response,
            type: err.type,
            error: const ApiException(
              code: 'B0001',
              message: '服务器内部错误，请稍后重试',
            ),
          );
        }
        return err;

      case DioExceptionType.cancel:
        return DioException(
          requestOptions: err.requestOptions,
          type: err.type,
          error: const ApiException(
            code: 'A0001',
            message: '请求已取消',
          ),
        );

      case DioExceptionType.badCertificate:
      case DioExceptionType.unknown:
        return DioException(
          requestOptions: err.requestOptions,
          type: err.type,
          error: ApiException(
            code: 'B0001',
            message: err.message ?? '发生未知错误',
          ),
        );
    }
  }
}
