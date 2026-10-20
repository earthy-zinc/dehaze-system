import 'package:clarity_flutter/core/network/api_result.dart';
import 'package:clarity_flutter/core/network/task_poller.dart';
import 'package:clarity_flutter/models/image_task_model.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('pollTask', () {
    test('已取消时立即抛出 PollCancelledException', () async {
      final cancellation = PollCancellation()..cancel();

      expect(
        () => pollTask<int>(
          (_) async => 1,
          99,
          statusOf: (_) => TaskStatus.processing,
          options: PollOptions(
            intervalMs: 1,
            timeoutMs: 50,
            cancellation: cancellation,
          ),
        ),
        throwsA(isA<PollCancelledException>()),
      );
    });

    test('轮询到终态即返回结果', () async {
      var calls = 0;

      final result = await pollTask<int>(
        (_) async => ++calls,
        1,
        statusOf: (value) =>
            value >= 2 ? TaskStatus.completed : TaskStatus.processing,
        options: const PollOptions(intervalMs: 1, timeoutMs: 200),
      );

      expect(result, 2);
      expect(calls, 2);
    });

    test('超时抛出 ApiException(B0100)', () async {
      await expectLater(
        pollTask<int>(
          (_) async => 1,
          7,
          statusOf: (_) => TaskStatus.processing,
          options: const PollOptions(intervalMs: 1, timeoutMs: 20),
        ),
        throwsA(isA<ApiException>().having((e) => e.code, 'code', 'B0100')),
      );
    });
  });
}
