import '../../models/prediction_model.dart';

/// 轮询配置
class PollOptions {
  const PollOptions({
    this.intervalMs = 2000,
    this.timeoutMs = 120000,
    this.onPoll,
  });

  /// 轮询间隔（毫秒）
  final int intervalMs;

  /// 最大等待时间（毫秒）
  final int timeoutMs;

  /// 每次轮询回调
  final void Function(TaskStatus status)? onPoll;
}

/// 轮询异步任务（POST 立即返回 status=processing 时，
/// 按间隔轮询 GET 直到 completed/failed 或超时）。
///
/// 预测与评估共用同一套轮询机制。
Future<T> pollTask<T>(
  Future<T> Function(int taskId) getStatus,
  int taskId, {
  required TaskStatus Function(T) statusOf,
  PollOptions? options,
}) async {
  final interval = options?.intervalMs ?? 2000;
  final timeout = options?.timeoutMs ?? 120000;
  final deadline = DateTime.now().add(Duration(milliseconds: timeout));

  while (DateTime.now().isBefore(deadline)) {
    await Future<void>.delayed(Duration(milliseconds: interval));
    final result = await getStatus(taskId);
    final status = statusOf(result);
    options?.onPoll?.call(status);
    if (status == TaskStatus.completed || status == TaskStatus.failed) {
      return result;
    }
  }
  throw Exception('任务 $taskId 超时（${timeout}ms）');
}
