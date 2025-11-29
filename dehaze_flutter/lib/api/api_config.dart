class ApiConfig {
  // 基础API URL - 你可以根据需要修改为你的服务器地址
  static const String _baseUrl = 'http://localhost:8080';
  static const String _apiVersion = 'api/v1';

  // 获取完整的API基础URL
  static String get baseUrl => _baseUrl;

  // 获取API版本前缀
  static String get apiVersion => _apiVersion;

  // 构建完整的API URL
  static String buildUrl(String endpoint) => '$_baseUrl/$_apiVersion$endpoint';

  // 请求超时配置
  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 60);
  static const Duration sendTimeout = Duration(seconds: 60);
}
