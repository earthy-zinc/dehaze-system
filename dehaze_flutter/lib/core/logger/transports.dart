import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:path_provider/path_provider.dart';

import '../network/api_config.dart';
import 'log_entry.dart';
import 'logger.dart';

/// transport 基类接口。
abstract class LogTransport {
  /// 逐条本地输出（不受采样/限流影响）。
  void log(LogEntry entry);

  /// 批量上报（仅生产 RemoteTransport 实现）。
  Future<void> send(List<LogEntry> logs) async {}
}

/// 开发环境 transport：输出到控制台（flutter console）。
class ConsoleTransport extends LogTransport {
  @override
  void log(LogEntry entry) {
    final tag = '[dehaze][${entry.level.label}]';
    final message = '${entry.message} trace_id=${entry.traceId ?? ""}';
    if (entry.level == LogLevel.error) {
      debugPrint('$tag $message\n${entry.errorStack ?? ""}');
    } else {
      debugPrint('$tag $message');
    }
  }
}

/// 本地文件 transport（开发+生产崩溃兜底）。
///
/// 目录结构 `logs/{yyyy-MM-dd}/{level}.log`（NDJSON），单文件 100MB 归档为
/// `{level}.{n}.log`，超过 [retentionDays] 的日期目录自动清理。
class FileTransport extends LogTransport {
  FileTransport({this.retentionDays = 7});

  /// 文件保留天数（开发 7 天，生产兜底 3 天）。
  final int retentionDays;

  static const int _maxFileBytes = 100 * 1024 * 1024; // 100MB

  String? _baseDir;
  bool _cleanedToday = false;

  @override
  void log(LogEntry entry) {
    if (_baseDir != null) {
      _writeSync(entry);
    } else {
      // 首次初始化目录（异步），初始化完成后写入缓存条目
      unawaited(_initBaseDir().then((_) => _writeSync(entry)));
    }
  }

  /// 写入一条日志到 info.log（全部级别）与 error.log（仅 ERROR）。
  void _writeSync(LogEntry entry) {
    try {
      final now = DateTime.now();
      final dateDir = _formatDate(now);
      final dir = Directory('$_baseDir/logs/$dateDir');
      dir.createSync(recursive: true);
      _cleanupOldDirs(dateDir);

      final line = '${entry.toNdjson()}\n';
      // info.log 包含 INFO+（全部级别）
      final infoTarget = _rotateIfNeeded(dir, 'info');
      infoTarget.writeAsStringSync(line, mode: FileMode.append);
      // error.log 仅 ERROR
      if (entry.level == LogLevel.error) {
        final errorTarget = _rotateIfNeeded(dir, 'error');
        errorTarget.writeAsStringSync(line, mode: FileMode.append);
      }
    } catch (_) {
      // 文件写入失败静默
    }
  }

  /// 获取应用文档目录路径。
  Future<void> _initBaseDir() async {
    if (_baseDir != null) return;
    final dir = await getApplicationDocumentsDirectory();
    _baseDir = dir.path;
  }

  String _formatDate(DateTime dt) {
    final m = dt.month.toString().padLeft(2, '0');
    final d = dt.day.toString().padLeft(2, '0');
    return '${dt.year}-$m-$d';
  }

  void _cleanupOldDirs(String currentDate) {
    if (_cleanedToday) return;
    _cleanedToday = true;
    try {
      final root = Directory('$_baseDir/logs');
      if (!root.existsSync()) return;
      final cutoff = DateTime.now().subtract(Duration(days: retentionDays));
      for (final dir in root.listSync().whereType<Directory>()) {
        final name = dir.uri.pathSegments.isNotEmpty
            ? dir.uri.pathSegments.last
            : '';
        final date = DateTime.tryParse(name);
        if (date != null && date.isBefore(cutoff)) {
          dir.deleteSync(recursive: true);
        }
      }
    } catch (_) {
      // 清理失败不阻塞
    }
  }

  File _rotateIfNeeded(Directory dir, String level) {
    final current = File('${dir.path}/$level.log');
    if (current.existsSync() && current.lengthSync() >= _maxFileBytes) {
      // 归档为 {level}.{n}.log（n 递增）
      var n = 1;
      while (File('${dir.path}/$level.$n.log').existsSync()) {
        n++;
      }
      current.renameSync('${dir.path}/$level.$n.log');
      return File('${dir.path}/$level.log');
    }
    return current;
  }

  /// 读取最近 error.log 中的日志条目（供启动补报使用）。
  Future<List<LogEntry>> readRecentErrorLogs({int limit = 50}) async {
    await _initBaseDir();
    final now = DateTime.now();
    final dateDir = _formatDate(now);
    final dir = Directory('$_baseDir/logs/$dateDir');
    final errorLog = File('${dir.path}/error.log');
    if (!errorLog.existsSync()) return const [];

    final lines = await errorLog.readAsLines();
    final start = lines.length > limit ? lines.length - limit : 0;
    final entries = <LogEntry>[];
    for (var i = start; i < lines.length; i++) {
      try {
        final json = jsonDecode(lines[i]) as Map<String, dynamic>;
        final level = json['level'] == 'ERROR'
            ? LogLevel.error
            : json['level'] == 'WARN'
                ? LogLevel.warn
                : LogLevel.info;
        entries.add(LogEntry(
          timestamp: json['timestamp']?.toString() ?? '',
          level: level,
          message: json['message']?.toString() ?? '',
          app: json['app']?.toString() ?? '',
          appVersion: json['app_version']?.toString() ?? '',
          url: json['url']?.toString(),
          userAgent: json['user_agent']?.toString(),
          traceId: json['trace_id']?.toString(),
          errorType: json['error_type']?.toString(),
          errorSource: json['error_source']?.toString(),
          errorStack: json['error_stack']?.toString(),
          method: json['method']?.toString(),
          path: json['path']?.toString(),
          status: json['status'] as int?,
          duration: (json['duration'] as num?)?.toDouble(),
          code: json['code']?.toString(),
        ));
      } catch (_) {
        // 单行解析失败跳过
      }
    }
    return entries;
  }
}

/// 生产环境 transport：批量上报后端接收 API。
class RemoteTransport implements LogTransport {
  RemoteTransport({Dio? dio}) : _dio = dio;

  final Dio? _dio;

  static const String _endpoint = '/logs/client';
  static const int _maxBatch = 50;

  @override
  void log(LogEntry entry) {
    // 生产环境不在控制台逐条刷屏
  }

  @override
  Future<void> send(List<LogEntry> logs) async {
    if (logs.isEmpty) return;
    final dio = _dio ?? Dio(BaseOptions(
      baseUrl: ApiConfig.apiBaseUrl,
      connectTimeout: const Duration(seconds: 15),
      receiveTimeout: const Duration(seconds: 15),
    ));
    final batch = logs.take(_maxBatch).toList();
    await dio.post<Map<String, dynamic>>(
      _endpoint,
      data: {
        'logs': batch.map((e) => e.toJson()).toList(),
      },
      options: Options(
        headers: {'Content-Type': 'application/json;charset=utf-8'},
      ),
    );
  }
}
