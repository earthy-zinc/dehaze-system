import '../../models/image_task_model.dart';
import 'api_result.dart';

/// 轮询取消信号
///
/// 由调用方持有并在需要时取消（如用户点击取消、离开页面）；
/// 轮询循环检测到取消后抛出 [PollCancelledException]。
class PollCancellation {
  bool _cancelled = false;

  bool get isCancelled => _cancelled;

  void cancel() => _cancelled = true;
}

/// 轮询被取消时抛出
class PollCancelledException implements Exception {
  const PollCancelledException(this.taskId);

  final int taskId;

  @override
  String toString() => '任务 $taskId 轮询已取消';
}

/// 轮询配置
class PollOptions {
  const PollOptions({
    this.intervalMs = 2000,
    this.timeoutMs = 120000,
    this.onPoll,
    this.cancellation,
  });

  /// 轮询间隔（毫秒）
  final int intervalMs;

  /// 最大等待时间（毫秒）
  final int timeoutMs;

  /// 每次轮询回调
  final void Function(TaskStatus status)? onPoll;

  /// 取消信号
  final PollCancellation? cancellation;
}

/// 轮询异步任务（POST 立即返回 status=processing 时，
/// 按间隔轮询 GET 直到终态、取消或超时）。
///
/// 图像任务与评估共用同一套轮询机制。
Future<T> pollTask<T>(
  Future<T> Function(int taskId) getStatus,
  int taskId, {
  required TaskStatus Function(T) statusOf,
  PollOptions? options,
}) async {
  final interval = options?.intervalMs ?? 2000;
  final timeout = options?.timeoutMs ?? 120000;
  final cancellation = options?.cancellation;
  final deadline = DateTime.now().add(Duration(milliseconds: timeout));

  while (DateTime.now().isBefore(deadline)) {
    if (cancellation?.isCancelled ?? false) {
      throw PollCancelledException(taskId);
    }
    await Future<void>.delayed(Duration(milliseconds: interval));
    final result = await getStatus(taskId);
    final status = statusOf(result);
    options?.onPoll?.call(status);
    if (status != TaskStatus.processing) {
      return result;
    }
  }
  throw ApiException(code: 'B0100', message: '任务处理超时，请稍后在任务历史中查看结果');
}
