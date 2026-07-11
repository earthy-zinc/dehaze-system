import 'dart:async';

import 'package:dio/dio.dart';

import '../../constants/api_constants.dart';
import '../../storage/token_storage.dart';
import '../api_result.dart';

/// 重试拦截器
///
/// 处理 Token 过期自动刷新 + 并发请求队列：
/// 1. 请求A收到 401/A0230 → 启动刷新流程
/// 2. 请求B/C/D 同时收到 401 → 加入等待队列
/// 3. 刷新成功 → 用新 Token 重发所有排队请求
/// 4. 刷新失败 → 清除 Token，触发 onAuthError，拒绝所有排队请求
class RetryInterceptor extends Interceptor {
  RetryInterceptor({
    required this.dio,
    required this.tokenStorage,
    this.onAuthError,
  });

  final Dio dio;
  final TokenStorage tokenStorage;
  final void Function()? onAuthError;

  /// 是否正在刷新 Token
  bool _isRefreshing = false;

  /// 等待刷新完成的请求队列
  final List<_RequestTask> _pendingQueue = [];

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    // 判断是否为 Token 过期错误
    if (!_isTokenExpiredError(err)) {
      handler.next(err);
      return;
    }

    // 刷新接口自身的错误不重试
    if (err.requestOptions.path.contains(ApiConstants.authRefresh)) {
      handler.next(err);
      return;
    }

    // 加入队列或启动刷新
    if (_isRefreshing) {
      // 已在刷新中，加入等待队列
      _pendingQueue.add(_RequestTask(
        requestOptions: err.requestOptions,
        handler: handler,
      ));
      return;
    }

    // 启动刷新流程
    _startRefresh(err.requestOptions, handler);
  }

  /// 判断是否为 Token 过期错误
  bool _isTokenExpiredError(DioException err) {
    // ApiException 中 code 为 A0230/A0231
    if (err.error is ApiException) {
      final apiError = err.error as ApiException;
      return apiError.code == 'A0230' || apiError.code == 'A0231';
    }
    // HTTP 401
    if (err.response?.statusCode == 401) {
      return true;
    }
    return false;
  }

  /// 启动 Token 刷新流程
  void _startRefresh(
    RequestOptions requestOptions,
    ErrorInterceptorHandler handler,
  ) {
    _isRefreshing = true;

    // 将当前请求加入队列
    _pendingQueue.add(_RequestTask(
      requestOptions: requestOptions,
      handler: handler,
    ));

    _doRefresh().then((success) {
      _isRefreshing = false;

      if (success) {
        // 刷新成功，重发所有排队请求
        _retryAllPending();
      } else {
        // 刷新失败，拒绝所有排队请求并触发登出
        _rejectAllPending();
        onAuthError?.call();
      }
    });
  }

  /// 执行 Token 刷新（直接发请求，不经过拦截器）
  Future<bool> _doRefresh() async {
    final refreshToken = tokenStorage.refreshToken;
    if (refreshToken == null || refreshToken.isEmpty) {
      return false;
    }

    try {
      final response = await dio.post<Map<String, dynamic>>(
        ApiConstants.authRefresh,
        data: {'refreshToken': refreshToken},
      );

      final data = response.data;
      if (data is Map<String, dynamic> &&
          data['code']?.toString() == '00000') {
        final tokenData = data['data'] as Map<String, dynamic>?;
        if (tokenData != null) {
          await tokenStorage.saveTokens(
            accessToken: tokenData['accessToken'] as String,
            refreshToken: tokenData['refreshToken'] as String?,
          );
          return true;
        }
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// 重发所有排队请求
  void _retryAllPending() {
    final queue = List<_RequestTask>.from(_pendingQueue);
    _pendingQueue.clear();

    for (final task in queue) {
      _retryRequest(task);
    }
  }

  /// 重发单个请求
  void _retryRequest(_RequestTask task) {
    final options = task.requestOptions;

    // 更新 Token
    final token = tokenStorage.accessToken;
    if (token != null && token.isNotEmpty) {
      options.headers['Authorization'] = 'Bearer $token';
    }

    // 使用原始 dio 重发
    dio
        .fetch<dynamic>(options)
        .then((response) => task.handler.resolve(response))
        .catchError((Object error) {
      if (error is DioException) {
        task.handler.reject(error);
      } else {
        task.handler.reject(
          DioException(
            requestOptions: options,
            error: error,
          ),
        );
      }
    });
  }

  /// 拒绝所有排队请求
  void _rejectAllPending() {
    final queue = List<_RequestTask>.from(_pendingQueue);
    _pendingQueue.clear();

    for (final task in queue) {
      task.handler.reject(
        DioException(
          requestOptions: task.requestOptions,
          error: const ApiException(
            code: 'A0230',
            message: '登录已过期，请重新登录',
          ),
        ),
      );
    }
  }
}

/// 排队请求任务
class _RequestTask {
  _RequestTask({required this.requestOptions, required this.handler});

  final RequestOptions requestOptions;
  final ErrorInterceptorHandler handler;
}
