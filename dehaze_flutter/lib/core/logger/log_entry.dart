import 'dart:convert';

import 'logger.dart';

/// 单条前端日志条目。
///
/// 字段规范见前端日志监控改造计划 §3.3（与 JS SDK 对齐，service=client）。
class LogEntry {
  const LogEntry({
    required this.timestamp,
    required this.level,
    required this.message,
    required this.app,
    required this.appVersion,
    this.url,
    this.userAgent,
    this.traceId,
    this.errorType,
    this.errorSource,
    this.errorStack,
    this.method,
    this.path,
    this.status,
    this.duration,
    this.code,
    this.dedupCount,
  });

  /// ISO8601 UTC 时间戳
  final String timestamp;

  /// 日志级别
  final LogLevel level;

  /// 人读描述（≤2000 字符）
  final String message;

  /// 前端项目标识，固定 'flutter'
  final String app;

  /// 应用版本号
  final String appVersion;

  /// 当前路由/页面
  final String? url;

  /// 设备 User-Agent
  final String? userAgent;

  /// 与后端日志串联的 trace_id
  final String? traceId;

  /// 错误类型：dart
  final String? errorType;

  /// 错误来源：FlutterError / Zone / api_interceptor
  final String? errorSource;

  /// 完整堆栈字符串（≤8000 字符）
  final String? errorStack;

  /// HTTP 方法（API 失败日志）
  final String? method;

  /// 请求路径（API 失败日志，不含 query）
  final String? path;

  /// HTTP 状态码
  final int? status;

  /// 请求耗时（毫秒）
  final double? duration;

  /// 业务错误码
  final String? code;

  /// ERROR 去重汇总标记：10s 窗口内相同 fingerprint 的总命中次数（仅汇总条目携带，>1）
  final int? dedupCount;

  /// 序列化为 API 上报/落盘的 JSON map（NDJSON）。
  ///
  /// 仅输出非空字段，避免冗余。
  Map<String, dynamic> toJson() {
    final map = <String, dynamic>{
      'timestamp': timestamp,
      'level': level.label,
      'message': message,
      'app': app,
      'app_version': appVersion,
    };
    if (url != null && url!.isNotEmpty) map['url'] = url;
    if (userAgent != null && userAgent!.isNotEmpty) map['user_agent'] = userAgent;
    if (traceId != null && traceId!.isNotEmpty) map['trace_id'] = traceId;
    if (errorType != null) map['error_type'] = errorType;
    if (errorSource != null) map['error_source'] = errorSource;
    if (errorStack != null) map['error_stack'] = errorStack;
    if (method != null) map['method'] = method;
    if (path != null) map['path'] = path;
    if (status != null) map['status'] = status;
    if (duration != null) map['duration'] = duration;
    if (code != null) map['code'] = code;
    if (dedupCount != null) map['dedup_count'] = dedupCount;
    return map;
  }

  /// 序列化为 NDJSON 单行。
  String toNdjson() => jsonEncode(toJson());
}
