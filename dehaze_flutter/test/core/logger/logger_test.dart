import 'dart:convert';

import 'package:dehaze_flutter/core/logger/log_entry.dart';
import 'package:dehaze_flutter/core/logger/logger.dart';
import 'package:dehaze_flutter/core/logger/transports.dart';
import 'package:fake_async/fake_async.dart';
import 'package:test/test.dart';

/// ERROR 去重 + 次数汇总测试。使用 fakeAsync 控制时间推进，验证 10s 窗口行为。
///
/// 注：使用 package:test 而非 flutter_test，绕过 Flutter 3.35.7 工具链
/// `package:flutter_test/flutter_test.yaml` 解析 bug。Logger 逻辑为纯 Dart，
/// 不依赖 Flutter binding，可独立测试。
///
/// Logger.init 在 fakeAsync 内部调用，确保 flush 定时器与去重定时器
/// 都被 fakeAsync 捕获，避免真实 timer 泄漏到测试外。
void main() {
  tearDown(() {
    Logger.reset();
  });

  test('相同 fingerprint 在 10s 窗口内只输出首条，重复被去重', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('RenderFlex overflowed',
          errorStack: 'stack-A', errorType: 'dart');
      logger.error('RenderFlex overflowed',
          errorStack: 'stack-A', errorType: 'dart');
      logger.error('RenderFlex overflowed',
          errorStack: 'stack-A', errorType: 'dart');

      final real = transport.logs.where((l) => l.dedupCount == null).toList();
      expect(real.length, 1);
      expect(real.first.message, 'RenderFlex overflowed');
    });
  });

  test('窗口结束时补发汇总条目，dedupCount 标记总次数，message 标注重复次数', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error(
        'RenderFlex overflowed',
        errorStack: 'stack-A',
        errorType: 'dart',
        errorSource: 'flutter_error',
      );
      for (var i = 0; i < 5; i++) {
        logger.error('RenderFlex overflowed',
            errorStack: 'stack-A', errorType: 'dart');
      }

      // 推进时间超过 10s 窗口，触发定时器补发汇总
      async.elapse(const Duration(seconds: 10, milliseconds: 1));

      final summaries =
          transport.logs.where((l) => l.dedupCount != null).toList();
      expect(summaries.length, 1);
      expect(summaries.first.dedupCount, 6);
      expect(summaries.first.message, 'RenderFlex overflowed (10s 内重复 5 次)');
      expect(summaries.first.errorStack, 'stack-A');
      expect(summaries.first.errorType, 'dart');
      expect(summaries.first.errorSource, 'flutter_error');
    });
  });

  test('单次命中无重复时不补发汇总，避免噪声', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('one-shot-error');
      async.elapse(const Duration(seconds: 10, milliseconds: 1));

      expect(transport.logs.length, 1);
      expect(transport.logs.where((l) => l.dedupCount != null), isEmpty);
    });
  });

  test('不同 fingerprint 不去重，各自独立输出', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('error-A', errorStack: 'stack-A');
      logger.error('error-B', errorStack: 'stack-B');

      final real = transport.logs.where((l) => l.dedupCount == null).toList();
      expect(real.length, 2);
    });
  });

  test('窗口过期后相同 fingerprint 视为新 burst，先补发上一轮汇总再输出新首条', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('recurring-error', errorStack: 'stack-A');
      logger.error('recurring-error', errorStack: 'stack-A'); // 重复 1 次

      // 推进时间超过窗口，再触发相同错误 → 视为新 burst
      async.elapse(const Duration(seconds: 10, milliseconds: 1));
      logger.error('recurring-error', errorStack: 'stack-A');

      final summaries =
          transport.logs.where((l) => l.dedupCount != null).toList();
      expect(summaries.length, 1);
      expect(summaries.first.dedupCount, 2);

      // 第一轮首条 + 第二轮首条
      final real = transport.logs.where((l) => l.dedupCount == null).toList();
      expect(real.length, 2);
    });
  });

  test('不同 fingerprint 到来时先补发上一轮汇总', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('error-A', errorStack: 'stack-A');
      logger.error('error-A', errorStack: 'stack-A');
      logger.error('error-A', errorStack: 'stack-A');
      // 不同 fingerprint 到来：先补发 A 的汇总，再输出 B
      logger.error('error-B', errorStack: 'stack-B');

      final summaries =
          transport.logs.where((l) => l.dedupCount != null).toList();
      expect(summaries.length, 1);
      expect(summaries.first.dedupCount, 3);
      expect(summaries.first.message, contains('error-A'));

      // A 首条 + B 首条
      final real = transport.logs.where((l) => l.dedupCount == null).toList();
      expect(real.length, 2);
    });
  });

  test('WARN/INFO 不参与去重', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.warn('same-warn');
      logger.warn('same-warn');
      logger.info('same-info');
      logger.info('same-info');

      // WARN/INFO 不去重，全部经 _emit 输出到 transport
      expect(transport.logs.length, 4);
    });
  });

  test('汇总条目经 _emit 输出，toJson 携带 dedup_count 字段', () {
    fakeAsync((async) {
      final transport = CaptureTransport();
      Logger.init(app: 'flutter', appVersion: '1.0.0', transports: [transport]);
      final logger = Logger.instance;
      logger.error('storm-error', errorStack: 'stack-A');
      for (var i = 0; i < 9; i++) {
        logger.error('storm-error', errorStack: 'stack-A');
      }
      async.elapse(const Duration(seconds: 10, milliseconds: 1));

      final summaries =
          transport.logs.where((l) => l.dedupCount != null).toList();
      expect(summaries.length, 1);
      expect(summaries.first.dedupCount, 10);

      final json = summaries.first.toJson();
      expect(json['dedup_count'], 10);
      // 序列化为 NDJSON 后字段存在
      final decoded =
          jsonDecode(summaries.first.toNdjson()) as Map<String, dynamic>;
      expect(decoded['dedup_count'], 10);
    });
  });
}

/// 仅捕获 log 输出、不批量上报的 transport（用于观测采样前的本地输出）。
class CaptureTransport extends LogTransport {
  final List<LogEntry> logs = [];

  @override
  void log(LogEntry entry) {
    logs.add(entry);
  }
}
