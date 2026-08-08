import 'package:json_annotation/json_annotation.dart';

part 'auth_model.g.dart';

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

  final String captchaKey;

  final String captchaCode;

  Map<String, dynamic> toJson() => _$LoginRequestToJson(this);
}

@JsonSerializable()
class LoginResponse {
  const LoginResponse({
    required this.sessionId,
    required this.user,
  });

  factory LoginResponse.fromJson(Map<String, dynamic> json) =>
      _$LoginResponseFromJson(json);

  final String sessionId;

  final LoginUser user;

  Map<String, dynamic> toJson() => _$LoginResponseToJson(this);
}

@JsonSerializable()
class LoginUser {
  const LoginUser({
    required this.id,
    required this.username,
    required this.nickname,
  });

  factory LoginUser.fromJson(Map<String, dynamic> json) =>
      _$LoginUserFromJson(json);

  final int id;
  final String username;
  final String nickname;

  Map<String, dynamic> toJson() => _$LoginUserToJson(this);
}

@JsonSerializable()
class CaptchaResponse {
  const CaptchaResponse({
    required this.captchaKey,
    required this.captchaBase64,
  });

  factory CaptchaResponse.fromJson(Map<String, dynamic> json) =>
      _$CaptchaResponseFromJson(json);

  final String captchaKey;

  final String captchaBase64;

  Map<String, dynamic> toJson() => _$CaptchaResponseToJson(this);
}
