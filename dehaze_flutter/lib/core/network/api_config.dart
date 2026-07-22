import 'dart:io' show Platform;

import 'package:flutter/foundation.dart' show kIsWeb, kReleaseMode;

/// API 基础配置
///
/// 管理后端服务器地址、超时时间等网络配置
/// 注意：Windows 开发环境必须使用 127.0.0.1 而非 localhost
class ApiConfig {
  const ApiConfig._();

  /// Backend server address (Java backend port 8989)
  ///
  /// 解析优先级：
  /// 1. `--dart-define=API_BASE_URL=...` 注入的地址（最高优先级，用于生产/CI）
  /// 2. 生产环境（`flutter run --release`）使用默认生产地址
  /// 3. 开发环境：Android 模拟器用 10.0.2.2，其余平台用 127.0.0.1
  static String get baseUrl {
    const env = String.fromEnvironment('API_BASE_URL');
    if (env.isNotEmpty) return env;
    if (kReleaseMode) return 'https://api.dehaze.example.com:8989';
    if (kIsWeb) return 'http://127.0.0.1:8989';
    if (Platform.isAndroid) return 'http://10.0.2.2:8989';
    return 'http://127.0.0.1:8989';
  }

  /// Dataset static file server address (port 9000)
  ///
  /// 解析优先级同 [baseUrl]：
  /// 1. `--dart-define=DATASET_BASE_URL=...`
  /// 2. 生产环境默认地址
  /// 3. 开发环境平台相关地址
  static String get datasetBaseUrl {
    const env = String.fromEnvironment('DATASET_BASE_URL');
    if (env.isNotEmpty) return env;
    if (kReleaseMode) return 'https://cdn.dehaze.example.com:9000';
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
