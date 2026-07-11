import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../storage/token_storage.dart';
import 'api_config.dart';
import 'interceptors/auth_interceptor.dart';
import 'interceptors/error_interceptor.dart';
import 'interceptors/response_interceptor.dart';
import 'interceptors/retry_interceptor.dart';

/// API 客户端
///
/// Dio 单例封装，统一管理网络请求配置和拦截器
class ApiClient {
  ApiClient._internal(this.dio);

  final Dio dio;

  /// 工厂方法：创建配置完整的 Dio 实例
  factory ApiClient.create({
    required TokenStorage tokenStorage,
    void Function()? onAuthError,
  }) {
    final dio = Dio(BaseOptions(
      baseUrl: ApiConfig.apiBaseUrl,
      connectTimeout: ApiConfig.connectTimeout,
      receiveTimeout: ApiConfig.receiveTimeout,
      sendTimeout: ApiConfig.sendTimeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ));

    // 按顺序添加拦截器：
    // 1. AuthInterceptor - 请求时注入 Token
    // 2. ResponseInterceptor - 响应时判断 code
    // 3. RetryInterceptor - 401 时刷新 Token 并重试
    // 4. ErrorInterceptor - 统一错误转换
    final retryInterceptor = RetryInterceptor(
      dio: dio,
      tokenStorage: tokenStorage,
      onAuthError: onAuthError,
    );

    dio.interceptors.addAll([
      AuthInterceptor(tokenStorage),
      ResponseInterceptor(),
      retryInterceptor,
      ErrorInterceptor(onAuthError: onAuthError),
    ]);

    if (kDebugMode) {
      dio.interceptors.add(LogInterceptor(
        requestBody: true,
        responseBody: true,
        requestHeader: false,
        responseHeader: false,
        error: true,
        logPrint: (obj) => debugPrint('[DIO] $obj'),
      ));
    }

    return ApiClient._internal(dio);
  }

  // ==================== 便捷请求方法 ====================

  /// GET 请求
  Future<T> get<T>(
    String path, {
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(dynamic json)? converter,
  }) async {
    final response = await dio.get<T>(
      path,
      queryParameters: queryParameters,
      options: options,
    );
    final data = (response.data as Map<String, dynamic>?)?['data'];
    if (converter != null && data != null) {
      return converter(data);
    }
    return data as T;
  }

  /// POST 请求
  Future<T> post<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(dynamic json)? converter,
  }) async {
    final response = await dio.post<T>(
      path,
      data: data,
      queryParameters: queryParameters,
      options: options,
    );
    final responseData = (response.data as Map<String, dynamic>?)?['data'];
    if (converter != null && responseData != null) {
      return converter(responseData);
    }
    return responseData as T;
  }

  /// PUT 请求
  Future<T> put<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(dynamic json)? converter,
  }) async {
    final response = await dio.put<T>(
      path,
      data: data,
      queryParameters: queryParameters,
      options: options,
    );
    final responseData = (response.data as Map<String, dynamic>?)?['data'];
    if (converter != null && responseData != null) {
      return converter(responseData);
    }
    return responseData as T;
  }

  /// DELETE 请求
  Future<T> delete<T>(
    String path, {
    dynamic data,
    Map<String, dynamic>? queryParameters,
    Options? options,
    T Function(dynamic json)? converter,
  }) async {
    final response = await dio.delete<T>(
      path,
      data: data,
      queryParameters: queryParameters,
      options: options,
    );
    final responseData = (response.data as Map<String, dynamic>?)?['data'];
    if (converter != null && responseData != null) {
      return converter(responseData);
    }
    return responseData as T;
  }
}
