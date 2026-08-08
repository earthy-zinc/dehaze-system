import 'dart:developer' as developer;

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../storage/token_storage.dart';
import 'api_config.dart';
import 'interceptors/auth_interceptor.dart';
import 'interceptors/error_interceptor.dart';
import 'interceptors/response_interceptor.dart';
import 'interceptors/trace_interceptor.dart';

class ApiClient {
  ApiClient._internal(this.dio);

  final Dio dio;

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

    dio.interceptors.addAll([
      TraceInterceptor(),
      AuthInterceptor(tokenStorage),
      ResponseInterceptor(),
      ErrorInterceptor(onAuthError: onAuthError),
    ]);

    if (kDebugMode) {
      dio.interceptors.add(LogInterceptor(
        requestBody: true,
        responseBody: true,
        requestHeader: false,
        responseHeader: false,
        error: true,
        logPrint: (obj) => developer.log('[DIO] $obj', name: 'dio'),
      ));
    }

    return ApiClient._internal(dio);
  }
}
