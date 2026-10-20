import 'package:flutter/foundation.dart'
    show kIsWeb, kReleaseMode, defaultTargetPlatform, TargetPlatform;

/// API 基础配置
///
/// 管理后端服务器地址、超时时间等网络配置
/// 注意：Windows 开发环境必须使用 127.0.0.1 而非 localhost
class ApiConfig {
  const ApiConfig._();

  /// Backend server address (Java backend port 8989)
  ///
  /// 解析优先级：
  /// 1. `--dart-define=API_BASE_URL=...` 注入的地址（生产 / CI 必须注入）
  /// 2. 开发环境：Android 模拟器用 10.0.2.2，其余平台用 127.0.0.1
  ///
  /// release 构建未注入即抛错，避免误连占位地址。
  static String get baseUrl {
    const env = String.fromEnvironment('API_BASE_URL');
    if (env.isNotEmpty) return env;
    if (kReleaseMode) {
      throw StateError('release 构建必须注入后端地址：--dart-define=API_BASE_URL=<url>');
    }
    if (kIsWeb) return 'http://127.0.0.1:8989';
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:8989';
    }
    return 'http://127.0.0.1:8989';
  }

  /// Dataset static file server address (port 19000)
  ///
  /// 解析优先级同 [baseUrl]。
  static String get datasetBaseUrl {
    const env = String.fromEnvironment('DATASET_BASE_URL');
    if (env.isNotEmpty) return env;
    if (kReleaseMode) {
      throw StateError(
        'release 构建必须注入数据集静态地址：--dart-define=DATASET_BASE_URL=<url>',
      );
    }
    if (kIsWeb) return 'http://127.0.0.1:19000';
    if (defaultTargetPlatform == TargetPlatform.android) {
      return 'http://10.0.2.2:19000';
    }
    return 'http://127.0.0.1:19000';
  }

  /// API 版本前缀
  static const String apiVersion = 'api/v1';

  /// 完整的 API 基础路径
  static String get apiBaseUrl => '$baseUrl/$apiVersion';

  // ==================== 超时配置 ====================

  static const Duration connectTimeout = Duration(seconds: 15);
  static const Duration receiveTimeout = Duration(seconds: 60);
  static const Duration sendTimeout = Duration(seconds: 60);
}
