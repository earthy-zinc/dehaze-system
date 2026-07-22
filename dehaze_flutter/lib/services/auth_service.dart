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
  /// Content-Type: application/x-www-form-urlencoded
  Future<LoginResponse> login(LoginRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.authLogin,
      data: request.toJson(),
      options: Options(
        headers: {'Content-Type': 'application/json'},
      ),
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return LoginResponse.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '登录失败');
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

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return CaptchaResponse.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取验证码失败');
  }

  /// 获取当前登录用户信息
  ///
  /// GET /auth/me
  Future<UserModel> getCurrentUser() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authMe,
    );

    final result = response.data!;
    if (result['code']?.toString() == ApiConstants.successCode) {
      return UserModel.fromJson(result['data'] as Map<String, dynamic>);
    }
    throw Exception(result['msg'] ?? '获取用户信息失败');
  }
}
