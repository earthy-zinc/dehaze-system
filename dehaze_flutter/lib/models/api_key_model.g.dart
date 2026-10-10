// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'api_key_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ApiKeyVO _$ApiKeyVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ApiKeyVO', json, ($checkedConvert) {
      final val = ApiKeyVO(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        keyPrefix: $checkedConvert('keyPrefix', (v) => v as String),
        status: $checkedConvert('status', (v) => (v as num).toInt()),
        apiKey: $checkedConvert('apiKey', (v) => v as String?),
        expiresAt: $checkedConvert('expiresAt', (v) => v as String?),
        lastUsedAt: $checkedConvert('lastUsedAt', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        dailyQuota: $checkedConvert('dailyQuota', (v) => (v as num?)?.toInt()),
        monthlyQuota: $checkedConvert(
          'monthlyQuota',
          (v) => (v as num?)?.toInt(),
        ),
        rpmLimit: $checkedConvert('rpmLimit', (v) => (v as num?)?.toInt()),
        modelWhitelist: $checkedConvert(
          'modelWhitelist',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ApiKeyVOToJson(ApiKeyVO instance) => <String, dynamic>{
  'id': instance.id,
  'name': instance.name,
  'apiKey': ?instance.apiKey,
  'keyPrefix': instance.keyPrefix,
  'status': instance.status,
  'expiresAt': ?instance.expiresAt,
  'lastUsedAt': ?instance.lastUsedAt,
  'createTime': ?instance.createTime,
  'dailyQuota': ?instance.dailyQuota,
  'monthlyQuota': ?instance.monthlyQuota,
  'rpmLimit': ?instance.rpmLimit,
  'modelWhitelist': ?instance.modelWhitelist,
};

ApiKeyCreateForm _$ApiKeyCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ApiKeyCreateForm', json, ($checkedConvert) {
      final val = ApiKeyCreateForm(
        name: $checkedConvert('name', (v) => v as String),
        expiresAt: $checkedConvert('expiresAt', (v) => v as String?),
        dailyQuota: $checkedConvert('dailyQuota', (v) => (v as num?)?.toInt()),
        monthlyQuota: $checkedConvert(
          'monthlyQuota',
          (v) => (v as num?)?.toInt(),
        ),
        rpmLimit: $checkedConvert('rpmLimit', (v) => (v as num?)?.toInt()),
        modelWhitelist: $checkedConvert(
          'modelWhitelist',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$ApiKeyCreateFormToJson(ApiKeyCreateForm instance) =>
    <String, dynamic>{
      'name': instance.name,
      'expiresAt': ?instance.expiresAt,
      'dailyQuota': ?instance.dailyQuota,
      'monthlyQuota': ?instance.monthlyQuota,
      'rpmLimit': ?instance.rpmLimit,
      'modelWhitelist': ?instance.modelWhitelist,
    };
