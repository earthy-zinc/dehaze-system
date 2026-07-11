import 'package:shared_preferences/shared_preferences.dart';

import '../constants/storage_constants.dart';

/// Token 持久化存储
///
/// 管理 JWT Token 的读写，基于 SharedPreferences
class TokenStorage {
  const TokenStorage(this._prefs);

  final SharedPreferences _prefs;

  /// 获取访问令牌
  String? get accessToken => _prefs.getString(StorageConstants.accessToken);

  /// 获取刷新令牌
  String? get refreshToken => _prefs.getString(StorageConstants.refreshToken);

  /// 是否有访问令牌
  bool get hasToken => accessToken != null && accessToken!.isNotEmpty;

  /// 是否有刷新令牌
  bool get hasRefreshToken =>
      refreshToken != null && refreshToken!.isNotEmpty;

  /// 保存令牌
  Future<void> saveTokens({
    required String accessToken,
    String? refreshToken,
  }) async {
    await _prefs.setString(StorageConstants.accessToken, accessToken);
    if (refreshToken != null) {
      await _prefs.setString(StorageConstants.refreshToken, refreshToken);
    }
  }

  /// 清除所有令牌
  Future<void> clearTokens() async {
    await _prefs.remove(StorageConstants.accessToken);
    await _prefs.remove(StorageConstants.refreshToken);
    await _prefs.remove(StorageConstants.userInfo);
  }
}
