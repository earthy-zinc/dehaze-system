/// API 基础配置
///
/// 管理后端服务器地址、超时时间等网络配置
/// 注意：Windows 开发环境必须使用 127.0.0.1 而非 localhost
class ApiConfig {
  const ApiConfig._();

  /// 后端服务地址（Java 后端端口 8989）
  static const String baseUrl = 'http://127.0.0.1:8989';

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
