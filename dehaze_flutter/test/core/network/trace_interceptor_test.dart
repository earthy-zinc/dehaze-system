import 'dart:convert';
import 'dart:typed_data';

import 'package:dehaze_flutter/core/logger/log_entry.dart';
import 'package:dehaze_flutter/core/logger/logger.dart';
import 'package:dehaze_flutter/core/logger/transports.dart';
import 'package:dehaze_flutter/core/network/interceptors/trace_interceptor.dart';
import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';

/// API 失败日志级别：401 是鉴权结果（未登录 / 会话失效），按 INFO 记录（不上报远端）；
/// 其余失败仍按 ERROR 上报，避免把预期结果混进真实故障。
void main() {
  tearDown(Logger.reset);

  test('401 记 INFO（API_UNAUTHORIZED），非 401 仍记 ERROR', () async {
    final transport = _CaptureTransport();
    Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);

    await _request(transport, 401, '/api/v1/auth/me');

    final unauthorized = transport.logs.single;
    expect(unauthorized.level, LogLevel.info);
    expect(unauthorized.message, 'API_UNAUTHORIZED');
    expect(unauthorized.method, 'GET');
    expect(unauthorized.path, '/api/v1/auth/me');
    expect(unauthorized.status, 401);

    await _request(transport, 500, '/api/v1/ai/providers');

    final failure = transport.logs.last;
    expect(failure.level, LogLevel.error);
    expect(failure.message, 'API_ERROR');
    expect(failure.errorSource, 'api_interceptor');
  });
}

Future<void> _request(
  _CaptureTransport transport,
  int status,
  String path,
) async {
  final dio = Dio()
    ..httpClientAdapter = _StatusAdapter(status)
    ..interceptors.add(TraceInterceptor());
  await expectLater(dio.get<dynamic>(path), throwsA(isA<DioException>()));
}

/// 固定返回指定状态码的适配器：不触达网络，只驱动拦截器的失败分支
class _StatusAdapter implements HttpClientAdapter {
  _StatusAdapter(this.status);

  final int status;

  @override
  Future<ResponseBody> fetch(
    RequestOptions options,
    Stream<Uint8List>? requestStream,
    Future<void>? cancelFuture,
  ) async {
    return ResponseBody.fromString(
      jsonEncode({'code': 'A0230', 'msg': '未登录或会话已失效'}),
      status,
      headers: {
        Headers.contentTypeHeader: [Headers.jsonContentType],
      },
    );
  }

  @override
  void close({bool force = false}) {}
}

/// 仅捕获 log 输出、不批量上报的 transport（观测采样前的本地输出）
class _CaptureTransport extends LogTransport {
  final List<LogEntry> logs = [];

  @override
  void log(LogEntry entry) => logs.add(entry);
}
