import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart' show kDebugMode, kReleaseMode, visibleForTesting;
import 'package:go_router/go_router.dart';

import 'log_entry.dart';
import 'transports.dart';

/// 日志级别（与后端字段规范对齐 §3.3.1）。
enum LogLevel {
  error('ERROR'),
  warn('WARN'),
  info('INFO');

  const LogLevel(this.label);
  final String label;
}

/// trace_id 生成与当前请求 trace 管理（与后端 §4.3 约定一致：hex 32 位无连字符）。
class Trace {
  Trace._();
  static final Random _random = Random.secure();
  static String _currentTraceId = '';

  static String generateTraceId() {
    final bytes = List<int>.generate(16, (_) => _random.nextInt(256));
    return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
  }

  static String get currentTraceId => _currentTraceId;
  static String ensureTraceId() {
    if (_currentTraceId.isEmpty) _currentTraceId = generateTraceId();
    return _currentTraceId;
  }

  static void alignTraceId(String? traceId) {
    if (traceId != null && traceId.isNotEmpty) _currentTraceId = traceId;
  }
}

/// Flutter 端日志 Logger 单例（多 transport 架构，§3.6）。
///
/// 采样限流（§3.4）：ERROR 100% / WARN 50% / INFO 不上报；60s 内最多 20 条。
/// 队列上限 500 条，满 10 条立即上报，30s 定时器，失败指数退避。
class Logger {
  Logger._();

  static Logger? _instance;

  /// 初始化 Logger 并返回单例。transports 未传时按构建模式自动组装默认列表。
  static Logger init({
    required String app,
    required String appVersion,
    List<LogTransport>? transports,
  }) {
    _instance ??= Logger._()
      .._app = app
      .._appVersion = appVersion
      .._transports = transports ??
          [
            ConsoleTransport(),
            if (kDebugMode) FileTransport(retentionDays: 7),
            if (kReleaseMode) FileTransport(retentionDays: 3),
            if (kReleaseMode) RemoteTransport(),
          ];
    _instance!._startFlushTimer();
    return _instance!;
  }

  static Logger get instance {
    if (_instance == null) {
      throw StateError('Logger not initialized. Call Logger.init() first.');
    }
    return _instance!;
  }

  static bool get isInitialized => _instance != null;

  late String _app;
  late String _appVersion;
  late List<LogTransport> _transports;

  /// 当前路由引用，用于在日志生成时自动填充 `url` 字段（ELK 按页面过滤）
  /// 在 runApp 后由调用方 attachRouter 注入；注入前发生的错误 url 为空
  GoRouter? _router;

  final List<LogEntry> _queue = [];
  bool _flushing = false;
  Timer? _flushTimer;
  Timer? _backoffTimer;
  int _backoffMs = 1000;
  final List<int> _sentTimestamps = [];

  // ERROR 去重：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条，
  // 窗口结束时若存在重复则补发一条汇总（dedupCount 标记总次数），避免日志风暴同时保留次数信息
  // （如 RenderFlex overflow 在 layout 阶段每帧抛出，60fps 下每秒 60 条相同日志）
  int _lastErrorFingerprint = 0;
  DateTime? _lastErrorTime;
  int _errorDedupCount = 0;
  LogEntry? _lastDedupEntry;
  Timer? _dedupSummaryTimer;
  static const Duration _dedupeWindow = Duration(seconds: 10);

  static const int _maxQueue = 500;
  static const int _flushThreshold = 10;
  static const Duration _flushInterval = Duration(seconds: 30);
  static const int _maxBackoffMs = 60000;
  static const int _rateLimitWindowMs = 60000;
  static const int _rateLimitMax = 20;
  static const int _maxMessageLength = 2000;
  static const int _maxStackLength = 8000;

  List<LogTransport> get transports => List.unmodifiable(_transports);
  List<FileTransport> get fileTransports =>
      _transports.whereType<FileTransport>().toList();

  /// 注入 GoRouter 引用，用于日志生成时自动取当前路由路径填充 `url` 字段。
  /// 在 runApp 后由根 Widget build 调用；注入前错误日志的 url 为空。
  void attachRouter(GoRouter router) => _router = router;

  /// 获取当前路由路径（matchedLocation），无 router 或获取失败返回 null。
  String? _currentRoutePath() {
    final router = _router;
    if (router == null) return null;
    try {
      final lastMatch = router.routerDelegate.currentConfiguration.last;
      return lastMatch.matchedLocation;
    } catch (_) {
      return null;
    }
  }

  /// 日志写入入口。
  void log(LogLevel level, String message, {
    String? url,
    String? traceId,
    String? errorType,
    String? errorSource,
    String? errorStack,
    String? method,
    String? path,
    int? status,
    double? duration,
    String? code,
  }) {
    final entry = LogEntry(
      timestamp: DateTime.now().toUtc().toIso8601String(),
      level: level,
      message: _truncate(message, _maxMessageLength),
      app: _app,
      appVersion: _appVersion,
      url: url ?? _currentRoutePath(),
      userAgent: 'Flutter/${Platform.operatingSystem}',
      traceId: traceId ?? Trace.currentTraceId,
      errorType: errorType,
      errorSource: errorSource,
      errorStack: errorStack != null ? _truncate(errorStack, _maxStackLength) : null,
      method: method,
      path: path,
      status: status,
      duration: duration,
      code: code,
    );

    // ERROR 去重：相同 fingerprint 在 10s 窗口内只输出首条，窗口结束时补发汇总
    if (level == LogLevel.error && _shouldDedupError(entry)) {
      return;
    }

    _emit(entry);
  }

  /// 实际输出日志条目：transport 输出 + 采样 + 限流 + 入队。去重汇总补发也走此路径
  void _emit(LogEntry entry) {
    for (final transport in _transports) {
      transport.log(entry);
    }

    // 采样：ERROR 100% / WARN 50% / INFO 不上报
    final rate = entry.level == LogLevel.error
        ? 100
        : (entry.level == LogLevel.warn ? 50 : 0);
    if (Random().nextDouble() * 100 > rate) return;
    if (!_allowReport()) return;
    _enqueue(entry);
  }

  /// ERROR 去重判定：相同 message + error_stack fingerprint 在 10s 窗口内只输出首条。
  /// 窗口内重复命中累加计数并跳过输出；新 fingerprint 或窗口过期时补发上一轮汇总。
  /// 返回 true 表示该条应被去重跳过，false 表示正常输出。
  bool _shouldDedupError(LogEntry entry) {
    final fingerprint = '${entry.message}|${entry.errorStack ?? ''}'.hashCode;
    final now = DateTime.now();
    final inWindow = _lastErrorTime != null &&
        now.difference(_lastErrorTime!) < _dedupeWindow;

    if (fingerprint == _lastErrorFingerprint && inWindow) {
      _errorDedupCount++;
      return true;
    }

    // 新 burst：先补发上一轮汇总（若有重复）
    _flushDedupSummary();
    _lastErrorFingerprint = fingerprint;
    _lastErrorTime = now;
    _errorDedupCount = 1;
    _lastDedupEntry = entry;
    _scheduleDedupSummary();
    return false;
  }

  /// 窗口结束时补发汇总条目：携带 dedupCount 标记本轮总次数，message 标注重复次数
  void _flushDedupSummary() {
    _dedupSummaryTimer?.cancel();
    _dedupSummaryTimer = null;
    final count = _errorDedupCount;
    final original = _lastDedupEntry;
    _errorDedupCount = 0;
    _lastErrorFingerprint = 0;
    _lastErrorTime = null;
    _lastDedupEntry = null;

    // 单次命中无重复时不补发，避免噪声
    if (count <= 1 || original == null) return;

    _emit(LogEntry(
      timestamp: DateTime.now().toUtc().toIso8601String(),
      level: LogLevel.error,
      message: _truncate(
        '${original.message} (10s 内重复 ${count - 1} 次)',
        _maxMessageLength,
      ),
      app: _app,
      appVersion: _appVersion,
      url: _currentRoutePath(),
      userAgent: 'Flutter/${Platform.operatingSystem}',
      traceId: Trace.currentTraceId,
      errorType: original.errorType,
      errorSource: original.errorSource,
      errorStack: original.errorStack,
      dedupCount: count,
    ));
  }

  void _scheduleDedupSummary() {
    _dedupSummaryTimer?.cancel();
    _dedupSummaryTimer = Timer(_dedupeWindow, _flushDedupSummary);
  }

  /// 重置单例与所有内部状态（定时器、队列、去重计数）。
  /// 仅供测试隔离使用，生产环境不应调用。
  @visibleForTesting
  static void reset() {
    final logger = _instance;
    if (logger == null) return;
    logger._flushTimer?.cancel();
    logger._flushTimer = null;
    logger._backoffTimer?.cancel();
    logger._backoffTimer = null;
    logger._dedupSummaryTimer?.cancel();
    logger._dedupSummaryTimer = null;
    logger._queue.clear();
    logger._sentTimestamps.clear();
    logger._errorDedupCount = 0;
    logger._lastErrorFingerprint = 0;
    logger._lastErrorTime = null;
    logger._lastDedupEntry = null;
    _instance = null;
  }

  void error(String message, {String? url, String? traceId, String? errorType,
    String? errorSource, String? errorStack, String? method, String? path,
    int? status, double? duration, String? code}) {
    log(LogLevel.error, message, url: url, traceId: traceId, errorType: errorType,
        errorSource: errorSource, errorStack: errorStack, method: method, path: path,
        status: status, duration: duration, code: code);
  }

  void warn(String message, {String? url, String? traceId, String? errorType,
    String? errorSource, String? errorStack, String? method, String? path,
    int? status, double? duration, String? code}) {
    log(LogLevel.warn, message, url: url, traceId: traceId, errorType: errorType,
        errorSource: errorSource, errorStack: errorStack, method: method, path: path,
        status: status, duration: duration, code: code);
  }

  void info(String message, {String? url, String? traceId}) {
    log(LogLevel.info, message, url: url, traceId: traceId);
  }

  bool _allowReport() {
    final now = DateTime.now().millisecondsSinceEpoch;
    while (_sentTimestamps.isNotEmpty && _sentTimestamps.first <= now - _rateLimitWindowMs) {
      _sentTimestamps.removeAt(0);
    }
    if (_sentTimestamps.length >= _rateLimitMax) return false;
    _sentTimestamps.add(now);
    return true;
  }

  void _enqueue(LogEntry entry) {
    if (_queue.length >= _maxQueue) _queue.removeAt(0);
    _queue.add(entry);
    if (_queue.length >= _flushThreshold) unawaited(flush());
  }

  Future<void> flush() async {
    if (_flushing || _queue.isEmpty) return;
    final remote = _transports.whereType<RemoteTransport>().firstOrNull;
    if (remote == null) return;

    _flushing = true;
    try {
      final batch = List<LogEntry>.from(_queue);
      _queue.clear();
      try {
        await remote.send(batch);
        _backoffMs = 1000;
      } catch (_) {
        _queue.insertAll(0, batch);
        _scheduleBackoff();
      }
    } finally {
      _flushing = false;
    }
  }

  void _scheduleBackoff() {
    if (_backoffTimer != null) return;
    final delay = _backoffMs;
    _backoffMs = min(_backoffMs * 2, _maxBackoffMs);
    _backoffTimer = Timer(Duration(milliseconds: delay), () {
      _backoffTimer = null;
      unawaited(flush());
    });
  }

  void _startFlushTimer() {
    _flushTimer ??= Timer.periodic(_flushInterval, (_) => unawaited(flush()));
  }

  /// App 进入后台时触发补报。
  void flushOnBackground() => unawaited(flush());

  /// 启动补报：从本地文件读取崩溃遗留日志并上报（§3.5）。
  /// 是否补报由调用方决定（如仅在 release 模式调用）。
  Future<void> flushFromDisk() async {
    final cached = <LogEntry>[];
    for (final ft in _transports.whereType<FileTransport>()) {
      cached.addAll(await ft.readRecentErrorLogs(limit: 50));
    }
    if (cached.isEmpty) return;
    _queue.addAll(cached);
    if (_queue.length > _maxQueue) {
      _queue.removeRange(0, _queue.length - _maxQueue);
    }
    await flush();
  }

  static String _truncate(String value, int max) =>
      value.length > max ? value.substring(0, max) : value;
}
