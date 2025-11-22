class AppConstants {
  // 应用信息
  static const String appName = '图像去雾应用';
  static const String appVersion = '1.0.0';
  static const String appBuildNumber = '1';

  // 网络配置
  static const Duration apiTimeout = Duration(seconds: 30);
  static const Duration connectTimeout = Duration(seconds: 10);
  static const Duration receiveTimeout = Duration(seconds: 25);
  static const Duration sendTimeout = Duration(seconds: 25);

  // 图像处理配置
  static const int maxImageSize = 10 * 1024 * 1024; // 10MB
  static const int maxImageWidth = 1920;
  static const int maxImageHeight = 1080;
  static const int defaultImageQuality = 85;

  // 支持的图像格式
  static const List<String> supportedImageFormats = [
    'jpg',
    'jpeg',
    'png',
    'webp',
    'bmp',
    'gif',
  ];

  // 缓存配置
  static const Duration cacheExpiration = Duration(hours: 24);
  static const int maxCacheSize = 100 * 1024 * 1024; // 100MB

  // UI 配置
  static const double defaultPadding = 16;
  static const double smallPadding = 8;
  static const double largePadding = 24;
  static const double borderRadius = 12;
  static const double smallBorderRadius = 8;

  // 动画配置
  static const Duration shortAnimation = Duration(milliseconds: 200);
  static const Duration mediumAnimation = Duration(milliseconds: 300);
  static const Duration longAnimation = Duration(milliseconds: 500);

  // 分页配置
  static const int defaultPageSize = 20;
  static const int maxPageSize = 100;

  // 文件上传配置
  static const int maxConcurrentUploads = 3;
  static const Duration uploadTimeout = Duration(minutes: 5);

  // 错误处理配置
  static const int maxRetryAttempts = 3;
  static const Duration retryDelay = Duration(seconds: 2);

  // 隐私和安全
  static const Duration sessionTimeout = Duration(hours: 24);
  static const int maxLoginAttempts = 5;
  static const Duration lockoutDuration = Duration(minutes: 15);

  // 性能监控
  static const Duration performanceReportInterval = Duration(minutes: 5);
  static const int maxLogEntries = 1000;

  // 开发配置
  static const bool enableDebugMode = true;
  static const bool enableLogging = true;
  static const bool enablePerformanceMonitoring = true;

  // 防抖和节流
  static const Duration searchDebounceDelay = Duration(milliseconds: 500);
  static const Duration buttonThrottleDelay = Duration(milliseconds: 1000);

  // 本地存储键
  static const String tokenKey = 'auth_token';
  static const String refreshTokenKey = 'refresh_token';
  static const String userProfileKey = 'user_profile';
  static const String settingsKey = 'app_settings';
  static const String themeKey = 'theme_mode';
  static const String languageKey = 'app_language';

  // 预设值
  static const List<String> defaultAlgorithmIds = [
    'ridcp',
    'dcp',
    'aod_net',
    'ffa_net',
    'dehamer',
  ];

  static const Map<String, String> algorithmDisplayNames = {
    'ridcp': 'RIDCP',
    'dcp': 'Dark Channel Prior',
    'aod_net': 'AOD-Net',
    'ffa_net': 'FFA-Net',
    'dehamer': 'Dehamer',
  };
}
