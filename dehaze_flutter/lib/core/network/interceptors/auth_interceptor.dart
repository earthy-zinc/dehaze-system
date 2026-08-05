import 'package:dio/dio.dart';

import '../../constants/api_constants.dart';
import '../../storage/token_storage.dart';

class AuthInterceptor extends Interceptor {
  AuthInterceptor(this._tokenStorage);

  final TokenStorage _tokenStorage;

  static const List<String> _publicPaths = [
    ApiConstants.authLogin,
    ApiConstants.authRegister,
    ApiConstants.authCaptcha,
  ];

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    final path = options.path;
    if (_isPublicPath(path)) {
      handler.next(options);
      return;
    }

    final sid = _tokenStorage.sessionId;
    if (sid != null && sid.isNotEmpty) {
      options.headers['X-Session-Id'] = sid;
    }

    handler.next(options);
  }

  bool _isPublicPath(String path) {
    for (final publicPath in _publicPaths) {
      if (path.contains(publicPath)) {
        return true;
      }
    }
    return false;
  }
}
