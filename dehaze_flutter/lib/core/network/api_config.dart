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

// API端点定义
class ApiEndpoints {
  // 图像去雾相关
  static const String dehaze = '/dehaze/process';
  static const String getStatus = '/dehaze/status';
  static const String cancelProcess = '/dehaze/cancel';
  static const String getAlgorithms = '/dehaze/algorithms';

  // 认证相关
  static const String login = '/auth/login';
  static const String register = '/auth/register';
  static const String refreshToken = '/auth/refresh';

  // 用户相关
  static const String getUserProfile = '/user/profile';
  static const String updateUserProfile = '/user/profile';

  // 图像历史相关
  static const String getHistory = '/history';
  static const String deleteImage = '/history';

  // 上传文件
  static const String uploadImage = '/upload';
}

// HTTP状态码
class HttpStatusCodes {
  static const int ok = 200;
  static const int created = 201;
  static const int noContent = 204;
  static const int badRequest = 400;
  static const int unauthorized = 401;
  static const int forbidden = 403;
  static const int notFound = 404;
  static const int methodNotAllowed = 405;
  static const int internalServerError = 500;
  static const int badGateway = 502;
  static const int serviceUnavailable = 503;
}
