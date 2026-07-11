// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'auth_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

LoginRequest _$LoginRequestFromJson(Map<String, dynamic> json) =>
    $checkedCreate(
      'LoginRequest',
      json,
      ($checkedConvert) {
        final val = LoginRequest(
          username: $checkedConvert('username', (v) => v as String),
          password: $checkedConvert('password', (v) => v as String),
          captchaId: $checkedConvert('captcha_id', (v) => v as String),
          captchaCode: $checkedConvert('captcha_code', (v) => v as String),
        );
        return val;
      },
      fieldKeyMap: const {
        'captchaId': 'captcha_id',
        'captchaCode': 'captcha_code',
      },
    );

Map<String, dynamic> _$LoginRequestToJson(LoginRequest instance) =>
    <String, dynamic>{
      'username': instance.username,
      'password': instance.password,
      'captcha_id': instance.captchaId,
      'captcha_code': instance.captchaCode,
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
        captchaId: $checkedConvert('captchaId', (v) => v as String),
        captchaImg: $checkedConvert('captchaImg', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$CaptchaResponseToJson(CaptchaResponse instance) =>
    <String, dynamic>{
      'captchaId': instance.captchaId,
      'captchaImg': instance.captchaImg,
    };
