import 'package:dio/dio.dart';

import '../../constants/api_constants.dart';
import '../../storage/token_storage.dart';

/// 认证拦截器
///
/// 在请求头中自动注入 JWT Token
/// 跳过公开接口（登录、验证码、刷新令牌）
class AuthInterceptor extends Interceptor {
  AuthInterceptor(this._tokenStorage);

  final TokenStorage _tokenStorage;

  /// 不需要认证的路径前缀
  static const List<String> _publicPaths = [
    ApiConstants.authLogin,
    ApiConstants.authCaptcha,
    ApiConstants.authRefresh,
  ];

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    // 公开接口不注入 Token
    final path = options.path;
    if (_isPublicPath(path)) {
      handler.next(options);
      return;
    }

    // 注入 Token
    final token = _tokenStorage.accessToken;
    if (token != null && token.isNotEmpty) {
      options.headers['Authorization'] = 'Bearer $token';
    }

    handler.next(options);
  }

  /// 判断是否为公开路径
  bool _isPublicPath(String path) {
    for (final publicPath in _publicPaths) {
      if (path.contains(publicPath)) {
        return true;
      }
    }
    return false;
  }
}
