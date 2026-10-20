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

LoginResponse _$LoginResponseFromJson(Map<String, dynamic> json) =>
    $checkedCreate('LoginResponse', json, ($checkedConvert) {
      final val = LoginResponse(
        sessionId: $checkedConvert('sessionId', (v) => v as String),
        user: $checkedConvert(
          'user',
          (v) => LoginUser.fromJson(v as Map<String, dynamic>),
        ),
      );
      return val;
    });

Map<String, dynamic> _$LoginResponseToJson(LoginResponse instance) =>
    <String, dynamic>{
      'sessionId': instance.sessionId,
      'user': instance.user.toJson(),
    };

LoginUser _$LoginUserFromJson(Map<String, dynamic> json) =>
    $checkedCreate('LoginUser', json, ($checkedConvert) {
      final val = LoginUser(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        username: $checkedConvert('username', (v) => v as String),
        nickname: $checkedConvert('nickname', (v) => v as String),
      );
      return val;
    });

Map<String, dynamic> _$LoginUserToJson(LoginUser instance) => <String, dynamic>{
  'id': instance.id,
  'username': instance.username,
  'nickname': instance.nickname,
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
