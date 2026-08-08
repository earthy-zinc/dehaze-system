import 'package:dio/dio.dart';

import '../../logger/logger.dart';

/// trace_id 透传 + API 失败日志上报拦截器。
///
/// - 请求拦截：生成/复用 trace_id 注入 `X-Trace-Id` 头
/// - 响应拦截：读取响应头 `X-Trace-Id` 与本地对齐
/// - 错误拦截：失败请求交 Logger 上报（关联 method/path/status/duration/code）
class TraceInterceptor extends Interceptor {
  final Map<String, int> _startTimes = {};

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    // 生成/复用 trace_id 注入请求头
    final traceId = Trace.ensureTraceId();
    options.headers['X-Trace-Id'] = traceId;
    _startTimes[options.uri.toString()] = DateTime.now().millisecondsSinceEpoch;
    handler.next(options);
  }

  @override
  void onResponse(Response<dynamic> response, ResponseInterceptorHandler handler) {
    // 读取响应头 X-Trace-Id 对齐
    final responseTraceId = response.headers.value('X-Trace-Id');
    if (responseTraceId != null && responseTraceId.isNotEmpty) {
      Trace.alignTraceId(responseTraceId);
    }
    _startTimes.remove(response.requestOptions.uri.toString());
    handler.next(response);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    final request = err.requestOptions;
    final status = err.response?.statusCode;
    final method = request.method.toUpperCase();
    final path = request.path;

    // 避免为日志上报 API 自身再记录（防循环）
    if (path.contains('/logs/client')) {
      handler.next(err);
      return;
    }

    // 提取业务错误码
    String? code;
    final data = err.response?.data;
    if (data is Map && data['code'] != null) {
      code = data['code'].toString();
    }

    Logger.instance.error(
      'API_ERROR',
      traceId: Trace.currentTraceId,
      method: method,
      path: path,
      status: status,
      duration: _durationMs(request),
      code: code,
      errorType: 'api',
      errorSource: 'api_interceptor',
      errorStack: err.toString(),
    );

    handler.next(err);
  }

  double? _durationMs(RequestOptions request) {
    final start = _startTimes.remove(request.uri.toString());
    if (start == null) return null;
    return (DateTime.now().millisecondsSinceEpoch - start).toDouble();
  }
}
