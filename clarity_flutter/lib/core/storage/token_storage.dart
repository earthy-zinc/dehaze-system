import 'package:shared_preferences/shared_preferences.dart';

import '../constants/storage_constants.dart';

class TokenStorage {
  const TokenStorage(this._prefs);

  final SharedPreferences _prefs;

  String? get sessionId => _prefs.getString(StorageConstants.sessionId);

  bool get hasToken => sessionId != null && sessionId!.isNotEmpty;

  Future<void> saveSessionId(String id) async {
    await _prefs.setString(StorageConstants.sessionId, id);
  }

  Future<void> clearTokens() async {
    await _prefs.remove(StorageConstants.sessionId);
    await _prefs.remove(StorageConstants.userInfo);
  }
}
