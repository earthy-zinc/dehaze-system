import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb;

/// API 基础配置
///
/// 管理后端服务器地址、超时时间等网络配置
/// 注意：Windows 开发环境必须使用 127.0.0.1 而非 localhost
class ApiConfig {
  const ApiConfig._();

  /// Backend server address (Java backend port 8989)
  ///
  /// Android emulator must use 10.0.2.2 to reach the host machine;
  /// other platforms (Windows/Web/iOS emulator) use 127.0.0.1.
  /// Resolved at runtime, so it is a getter, not a const.
  static String get baseUrl {
    if (kIsWeb) return 'http://127.0.0.1:8989';
    if (Platform.isAndroid) return 'http://10.0.2.2:8989';
    return 'http://127.0.0.1:8989';
  }

  /// Dataset static file server address (port 9000)
  ///
  /// Platform-dependent, same rule as [baseUrl].
  static String get datasetBaseUrl {
    if (kIsWeb) return 'http://127.0.0.1:9000';
    if (Platform.isAndroid) return 'http://10.0.2.2:9000';
    return 'http://127.0.0.1:9000';
  }

  /// API 版本前缀
  static const String apiVersion = 'api/v1';

  /// 完整的 API 基础路径
  static String get apiBaseUrl => '$baseUrl/$apiVersion';

  // ==================== 超时配置 ====================

  static const Duration connectTimeout = Duration(seconds: 15);
  static const Duration receiveTimeout = Duration(seconds: 60);
  static const Duration sendTimeout = Duration(seconds: 60);

  // ==================== 文件上传限制 ====================

  /// 最大文件大小（20MB）
  static const int maxFileSize = 20 * 1024 * 1024;

  /// 支持的图片格式
  static const List<String> supportedImageFormats = [
    'jpg',
    'jpeg',
    'png',
    'webp',
    'bmp',
  ];
}
