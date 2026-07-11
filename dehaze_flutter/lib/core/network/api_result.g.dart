// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'api_result.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ApiResult<T> _$ApiResultFromJson<T>(
  Map<String, dynamic> json,
  T Function(Object? json) fromJsonT,
) => $checkedCreate('ApiResult', json, ($checkedConvert) {
  final val = ApiResult<T>(
    code: $checkedConvert('code', (v) => v as String? ?? ''),
    msg: $checkedConvert('msg', (v) => v as String? ?? ''),
    data: $checkedConvert(
      'data',
      (v) => _$nullableGenericFromJson(v, fromJsonT),
    ),
    traceId: $checkedConvert('trace_id', (v) => v as String?),
    timestamp: $checkedConvert('timestamp', (v) => (v as num?)?.toInt()),
    errors: $checkedConvert(
      'errors',
      (v) => (v as List<dynamic>?)
          ?.map((e) => ApiFieldError.fromJson(e as Map<String, dynamic>))
          .toList(),
    ),
  );
  return val;
}, fieldKeyMap: const {'traceId': 'trace_id'});

Map<String, dynamic> _$ApiResultToJson<T>(
  ApiResult<T> instance,
  Object? Function(T value) toJsonT,
) => <String, dynamic>{
  'code': instance.code,
  'msg': instance.msg,
  if (_$nullableGenericToJson(instance.data, toJsonT) case final value?)
    'data': value,
  if (instance.traceId case final value?) 'trace_id': value,
  if (instance.timestamp case final value?) 'timestamp': value,
  if (instance.errors?.map((e) => e.toJson()).toList() case final value?)
    'errors': value,
};

T? _$nullableGenericFromJson<T>(
  Object? input,
  T Function(Object? json) fromJson,
) => input == null ? null : fromJson(input);

Object? _$nullableGenericToJson<T>(
  T? input,
  Object? Function(T value) toJson,
) => input == null ? null : toJson(input);

ApiFieldError _$ApiFieldErrorFromJson(Map<String, dynamic> json) =>
    $checkedCreate('ApiFieldError', json, ($checkedConvert) {
      final val = ApiFieldError(
        field: $checkedConvert('field', (v) => v as String?),
        message: $checkedConvert('message', (v) => v as String?),
        code: $checkedConvert('code', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$ApiFieldErrorToJson(ApiFieldError instance) =>
    <String, dynamic>{
      if (instance.field case final value?) 'field': value,
      if (instance.message case final value?) 'message': value,
      if (instance.code case final value?) 'code': value,
    };
