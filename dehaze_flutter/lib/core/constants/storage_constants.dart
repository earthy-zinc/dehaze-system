/// 本地存储 Key 常量
///
/// 统一管理 SharedPreferences 存储键名
class StorageConstants {
  const StorageConstants._();

  /// 访问令牌
  static const String accessToken = 'access_token';

  /// 刷新令牌
  static const String refreshToken = 'refresh_token';

  /// 用户信息 JSON
  static const String userInfo = 'user_info';

  /// 主题模式 (light/dark/system)
  static const String themeMode = 'theme_mode';

  /// 首次启动标记
  static const String isFirstLaunch = 'is_first_launch';
}
