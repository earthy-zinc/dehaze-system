import 'package:json_annotation/json_annotation.dart';

part 'auth_model.g.dart';

/// 登录请求
@JsonSerializable()
class LoginRequest {
  const LoginRequest({
    required this.username,
    required this.password,
    required this.captchaKey,
    required this.captchaCode,
  });

  factory LoginRequest.fromJson(Map<String, dynamic> json) =>
      _$LoginRequestFromJson(json);

  final String username;
  final String password;

  /// 验证码 Key（后端返回的 captchaKey）
  @JsonKey(name: 'captchaKey')
  final String captchaKey;

  /// 用户输入的验证码
  @JsonKey(name: 'captchaCode')
  final String captchaCode;

  Map<String, dynamic> toJson() => _$LoginRequestToJson(this);
}

/// 登录响应
@JsonSerializable()
class LoginResponse {
  const LoginResponse({
    required this.accessToken,
    required this.tokenType,
    required this.expires,
    this.refreshToken,
  });

  factory LoginResponse.fromJson(Map<String, dynamic> json) =>
      _$LoginResponseFromJson(json);

  @JsonKey(name: 'accessToken')
  final String accessToken;

  @JsonKey(name: 'tokenType', defaultValue: 'Bearer')
  final String tokenType;

  /// 过期时间（秒）
  @JsonKey(defaultValue: 3600)
  final int expires;

  @JsonKey(name: 'refreshToken')
  final String? refreshToken;

  Map<String, dynamic> toJson() => _$LoginResponseToJson(this);
}

/// 验证码响应
@JsonSerializable()
class CaptchaResponse {
  const CaptchaResponse({
    required this.captchaKey,
    required this.captchaBase64,
  });

  factory CaptchaResponse.fromJson(Map<String, dynamic> json) =>
      _$CaptchaResponseFromJson(json);

  /// 验证码 Key（登录时需回传给后端）
  @JsonKey(name: 'captchaKey')
  final String captchaKey;

  /// 验证码图片（Base64 编码，可直接用于 Image.memory 或 base64 解码）
  @JsonKey(name: 'captchaBase64')
  final String captchaBase64;

  Map<String, dynamic> toJson() => _$CaptchaResponseToJson(this);
}
