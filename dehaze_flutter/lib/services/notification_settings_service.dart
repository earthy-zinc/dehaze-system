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
  Future<NotificationSettings> updateSettings(
    NotificationSettingsForm form,
  ) async {
    final response = await _dio.put<Map<String, dynamic>>(
      ApiConstants.notificationSettings,
      data: form.toJson(),
    );
    return NotificationSettings.fromJson(
      response.data!['data'] as Map<String, dynamic>,
    );
  }
}
