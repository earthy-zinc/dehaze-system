// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'auth_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LoginRequest _$LoginRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate('LoginRequest', json, ($checkedConvert) {
      final val = LoginRequest(
        username: $checkedConvert('username', (v) => v as String),
        password: $checkedConvert('password', (v) => v as String),
        captchaKey: $checkedConvert('captchaKey', (v) => v as String),
        captchaCode: $checkedConvert('captchaCode', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$LoginRequestToJson(LoginRequest instance) =>
    <String, dynamic>{
      'username': instance.username,
      'password': instance.password,
      'captchaKey': instance.captchaKey,
      'captchaCode': instance.captchaCode,
    };

LoginResponse _$LoginResponseFromJson(
  Map<String, dynamic> json,
) => $checkedCreate('LoginResponse', json, ($checkedConvert) {
  final val = LoginResponse(
    accessToken: $checkedConvert('accessToken', (v) => v as String),
    tokenType: $checkedConvert('tokenType', (v) => v as String? ?? 'Bearer'),
    expires: $checkedConvert('expires', (v) => (v as num?)?.toInt() ?? 3600),
    refreshToken: $checkedConvert('refreshToken', (v) => v as String?),
  );
  return val;
});

Map<String, dynamic> _$LoginResponseToJson(LoginResponse instance) =>
    <String, dynamic>{
      'accessToken': instance.accessToken,
      'tokenType': instance.tokenType,
      'expires': instance.expires,
      if (instance.refreshToken case final value?) 'refreshToken': value,
    };

CaptchaResponse _$CaptchaResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('CaptchaResponse', json, ($checkedConvert) {
      final val = CaptchaResponse(
        captchaKey: $checkedConvert('captchaKey', (v) => v as String),
        captchaBase64: $checkedConvert('captchaBase64', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$CaptchaResponseToJson(CaptchaResponse instance) =>
    <String, dynamic>{
      'captchaKey': instance.captchaKey,
      'captchaBase64': instance.captchaBase64,
    };
