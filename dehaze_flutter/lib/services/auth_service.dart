import 'package:dio/dio.dart';

import '../core/constants/api_constants.dart';
import '../models/auth_model.dart';
import '../models/user_model.dart';

class AuthService {
  const AuthService(this._dio);

  final Dio _dio;

  Future<LoginResponse> login(LoginRequest request) async {
    final response = await _dio.post<Map<String, dynamic>>(
      ApiConstants.authLogin,
      data: request.toJson(),
      options: Options(
        headers: {'Content-Type': 'application/json'},
      ),
    );
    return LoginResponse.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  Future<void> logout() async {
    await _dio.post<Map<String, dynamic>>(ApiConstants.authLogout);
  }

  Future<CaptchaResponse> getCaptcha() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authCaptcha,
    );
    return CaptchaResponse.fromJson(response.data!['data'] as Map<String, dynamic>);
  }

  Future<UserModel> getCurrentUser() async {
    final response = await _dio.get<Map<String, dynamic>>(
      ApiConstants.authMe,
    );
    return UserModel.fromJson(response.data!['data'] as Map<String, dynamic>);
  }
}
