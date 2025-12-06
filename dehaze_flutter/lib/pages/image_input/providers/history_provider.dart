import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/providers.dart';
import '../models/image_input_model.dart';
import '../services/history_service.dart';

// ==================== 服务 Provider ====================

/// 历史记录服务 Provider
final historyServiceProvider = Provider<HistoryService>((ref) {
  final prefs = ref.watch(sharedPreferencesProvider);
  return HistoryService(prefs);
});

// ==================== 历史记录 Notifier ====================

/// 历史记录状态管理
class HistoryNotifier extends StateNotifier<List<HistoryRecordModel>> {
  HistoryNotifier(this._service) : super([]) {
    loadHistory();
  }

  final HistoryService _service;

  /// 加载历史记录
  void loadHistory() {
    state = _service.getHistory();
  }

  /// 添加记录
  Future<void> addRecord(HistoryRecordModel record) async {
    await _service.saveRecord(record);
    loadHistory();
  }

  /// 删除单条记录
  Future<void> deleteRecord(String id) async {
    await _service.deleteRecord(id);
    loadHistory();
  }

  /// 清空所有记录
  Future<void> clearAll() async {
    await _service.clearHistory();
    state = [];
  }
}

/// 历史记录 Provider
final historyProvider =
    StateNotifierProvider<HistoryNotifier, List<HistoryRecordModel>>((ref) {
  final service = ref.watch(historyServiceProvider);
  return HistoryNotifier(service);
});

/// 分组后的历史记录
final groupedHistoryProvider = Provider<Map<String, List<HistoryRecordModel>>>((ref) {
  final history = ref.watch(historyProvider);

  if (history.isEmpty) {
    return {};
  }

  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final yesterday = today.subtract(const Duration(days: 1));
  final weekAgo = today.subtract(const Duration(days: 7));

  final grouped = <String, List<HistoryRecordModel>>{
    '今天': [],
    '昨天': [],
    '最近7天': [],
    '更早': [],
  };

  for (final record in history) {
    final recordDate = DateTime(
      record.timestamp.year,
      record.timestamp.month,
      record.timestamp.day,
    );

    if (recordDate == today) {
      grouped['今天']!.add(record);
    } else if (recordDate == yesterday) {
      grouped['昨天']!.add(record);
    } else if (recordDate.isAfter(weekAgo)) {
      grouped['最近7天']!.add(record);
    } else {
      grouped['更早']!.add(record);
    }
  }

  // 移除空分组
  grouped.removeWhere((key, value) => value.isEmpty);

  return grouped;
});

/// 历史记录数量
final historyCountProvider = Provider<int>((ref) {
  return ref.watch(historyProvider).length;
});
