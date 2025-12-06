import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import '../models/image_input_model.dart';

/// 历史记录服务
///
/// 使用 SharedPreferences 进行本地存储
class HistoryService {
  const HistoryService(this._prefs);

  final SharedPreferences _prefs;

  static const String _storageKey = 'dehaze_history';
  static const int maxRecords = 20;

  /// 获取所有历史记录
  List<HistoryRecordModel> getHistory() {
    try {
      final jsonString = _prefs.getString(_storageKey);
      if (jsonString == null || jsonString.isEmpty) {
        return [];
      }

      final List<dynamic> jsonList = json.decode(jsonString) as List<dynamic>;
      return jsonList
          .map((e) => HistoryRecordModel.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (e) {
      return [];
    }
  }

  /// 保存历史记录
  Future<void> saveRecord(HistoryRecordModel record) async {
    final history = getHistory();

    // 添加到列表开头
    history.insert(0, record);

    // 限制记录数量
    if (history.length > maxRecords) {
      history.removeRange(maxRecords, history.length);
    }

    await _saveHistory(history);
  }

  /// 删除单条记录
  Future<void> deleteRecord(String id) async {
    final history = getHistory();
    history.removeWhere((record) => record.id == id);
    await _saveHistory(history);
  }

  /// 清空所有历史记录
  Future<void> clearHistory() async {
    await _prefs.remove(_storageKey);
  }

  /// 保存历史记录列表
  Future<void> _saveHistory(List<HistoryRecordModel> history) async {
    final jsonList = history.map((e) => e.toJson()).toList();
    final jsonString = json.encode(jsonList);
    await _prefs.setString(_storageKey, jsonString);
  }

  /// 获取分组后的历史记录
  Map<String, List<HistoryRecordModel>> getGroupedHistory() {
    final history = getHistory();
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
  }
}
