// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'api_key_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ApiKeyVO _$ApiKeyVOFromJson(Map<String, dynamic> json) => $checkedCreate(
  'ApiKeyVO',
  json,
  ($checkedConvert) {
    final val = ApiKeyVO(
      id: $checkedConvert('id', (v) => (v as num).toInt()),
      userId: $checkedConvert('userId', (v) => (v as num?)?.toInt()),
      name: $checkedConvert('name', (v) => v as String?),
      keyPrefix: $checkedConvert('keyPrefix', (v) => v as String?),
      apiKey: $checkedConvert('apiKey', (v) => v as String?),
      permissions: $checkedConvert(
        'permissions',
        (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
      ),
      status: $checkedConvert('status', (v) => (v as num).toInt()),
      statusName: $checkedConvert('statusName', (v) => v as String?),
      lastUsedTime: $checkedConvert('lastUsedAt', (v) => v as String?),
      expireTime: $checkedConvert('expiresAt', (v) => v as String?),
      createTime: $checkedConvert('createTime', (v) => v as String?),
    );
    return val;
  },
  fieldKeyMap: const {'lastUsedTime': 'lastUsedAt', 'expireTime': 'expiresAt'},
);

Map<String, dynamic> _$ApiKeyVOToJson(ApiKeyVO instance) => <String, dynamic>{
  'id': instance.id,
  if (instance.userId case final value?) 'userId': value,
  if (instance.name case final value?) 'name': value,
  if (instance.keyPrefix case final value?) 'keyPrefix': value,
  if (instance.apiKey case final value?) 'apiKey': value,
  'permissions': instance.permissions,
  'status': instance.status,
  if (instance.statusName case final value?) 'statusName': value,
  if (instance.lastUsedTime case final value?) 'lastUsedAt': value,
  if (instance.expireTime case final value?) 'expiresAt': value,
  if (instance.createTime case final value?) 'createTime': value,
};

ApiKeyQuery _$ApiKeyQueryFromJson(Map<String, dynamic> json) => $checkedCreate(
  'ApiKeyQuery',
  json,
  ($checkedConvert) {
    final val = ApiKeyQuery(
      pageNum: $checkedConvert('pageNum', (v) => (v as num?)?.toInt() ?? 1),
      pageSize: $checkedConvert('pageSize', (v) => (v as num?)?.toInt() ?? 10),
      status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      keyword: $checkedConvert('keyword', (v) => v as String?),
    );
    return val;
  },
);

Map<String, dynamic> _$ApiKeyQueryToJson(ApiKeyQuery instance) =>
    <String, dynamic>{
      'pageNum': instance.pageNum,
      'pageSize': instance.pageSize,
      if (instance.status case final value?) 'status': value,
      if (instance.keyword case final value?) 'keyword': value,
    };

ApiKeyCreateForm _$ApiKeyCreateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ApiKeyCreateForm', json, ($checkedConvert) {
      final val = ApiKeyCreateForm(
        name: $checkedConvert('name', (v) => v as String),
        permissions: $checkedConvert(
          'permissions',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList() ?? [],
        ),
        expireTime: $checkedConvert('expiresAt', (v) => v as String?),
      );
      return val;
    }, fieldKeyMap: const {'expireTime': 'expiresAt'});

Map<String, dynamic> _$ApiKeyCreateFormToJson(ApiKeyCreateForm instance) =>
    <String, dynamic>{
      'name': instance.name,
      'permissions': instance.permissions,
      if (instance.expireTime case final value?) 'expiresAt': value,
    };

ApiKeyUpdateForm _$ApiKeyUpdateFormFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ApiKeyUpdateForm', json, ($checkedConvert) {
      final val = ApiKeyUpdateForm(
        name: $checkedConvert('name', (v) => v as String?),
        permissions: $checkedConvert(
          'permissions',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
        status: $checkedConvert('status', (v) => (v as num?)?.toInt()),
      );
      return val;
    });

Map<String, dynamic> _$ApiKeyUpdateFormToJson(ApiKeyUpdateForm instance) =>
    <String, dynamic>{
      if (instance.name case final value?) 'name': value,
      if (instance.permissions case final value?) 'permissions': value,
      if (instance.status case final value?) 'status': value,
    };
