import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/auth_model.dart';
import '../models/user_model.dart';

/// 认证服务
///
/// 封装认证相关 API 调用：
/// - login: 登录
/// - logout: 登出
/// - getCaptcha: 获取验证码
/// - refreshToken: 刷新令牌
/// - getCurrentUser: 获取当前用户信息
class AuthService {
  const AuthService(this._dio);

  final Dio _dio;

  /// 登录
  ///
  /// POST /auth/login
  Future<LoginResponse> login(LoginRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.authLogin,
      data: request.toJson(),
      options: Options(
        headers: {'Content-Type': 'application/json'},
      ),
    );
    // ResponseInterceptor 已保证 code=='00000'，失败已 reject 为 ApiException
    return LoginResponse.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 登出
  ///
  /// POST /auth/logout
  Future<void> logout() async {
    await _dio.post<Map<String, dynamic>>(ApiConstants.authLogout);
  }

  /// 获取验证码
  ///
  /// GET /auth/captcha
  Future<CaptchaResponse> getCaptcha() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authCaptcha,
    );
    return CaptchaResponse.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  /// 获取当前登录用户信息
  ///
  /// GET /auth/me
  Future<UserModel> getCurrentUser() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authMe,
    );
    return UserModel.fromJson(response.data!['data'] as Map<String, dynamic>);
  }
}
