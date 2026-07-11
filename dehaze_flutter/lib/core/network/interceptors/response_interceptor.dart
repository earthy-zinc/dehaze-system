import 'package:dio/dio.dart';

import '../api_result.dart';

/// 响应拦截器
///
/// 统一处理后端响应：
/// - code == "00000" → 正常返回 data
/// - code != "00000" → 抛出 ApiException
/// - HTTP 401 → 由 error_interceptor 处理
class ResponseInterceptor extends Interceptor {
  @override
  void onResponse(Response<dynamic> response, ResponseInterceptorHandler handler) {
    final data = response.data;

    // 非 JSON 响应（如文件下载），直接放行
    if (data is! Map<String, dynamic>) {
      handler.next(response);
      return;
    }

    final code = data['code']?.toString() ?? '';
    final msg = data['msg']?.toString() ?? '未知错误';

    // 业务成功
    if (code == '00000') {
      handler.next(response);
      return;
    }

    // 业务失败 → 抛出 ApiException
    final errorsJson = data['errors'] as List<dynamic>?;
    final errors = errorsJson
        ?.map((e) => ApiFieldError(
              field: e['field']?.toString(),
              message: e['message']?.toString(),
              code: e['code']?.toString(),
            ))
        .toList();

    handler.reject(
      DioException(
        requestOptions: response.requestOptions,
        response: response,
        type: DioExceptionType.badResponse,
        error: ApiException(
          code: code,
          message: msg,
          errors: errors,
        ),
      ),
    );
  }
}
