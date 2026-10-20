import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/notification_settings_model.dart';

class NotificationSettingsService {
  const NotificationSettingsService(this._dio);

  final Dio _dio;

  /// 获取当前用户通知设置
  Future<NotificationSettings> getSettings() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.notificationSettings,
    );
    return NotificationSettings.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }

  /// 更新通知设置
  ///
  /// 后端 PATCH 语义为 preferences 逐 key 深合并，调用方应提交完整偏好以避免丢字段。
  Future<NotificationSettings> updateSettings(
    NotificationSettings settings,
  ) async {
    final response = await _dio.patch<Map<String, dynamic>>(
      ApiConstants.notificationSettings,
      data: settings.toJson(),
    );
    return NotificationSettings.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
