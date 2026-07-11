// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'page_result.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

PageResult<T> _$PageResultFromJson<T>(
  Map<String, dynamic> json,
  T Function(Object? json) fromJsonT,
) => $checkedCreate('PageResult', json, ($checkedConvert) {
  final val = PageResult<T>(
    list: $checkedConvert(
      'list',
      (v) => (v as List<dynamic>?)?.map(fromJsonT).toList() ?? [],
    ),
    total: $checkedConvert('total', (v) => (v as num?)?.toInt() ?? 0),
  );
  return val;
});

Map<String, dynamic> _$PageResultToJson<T>(
  PageResult<T> instance,
  Object? Function(T value) toJsonT,
) => <String, dynamic>{
  'list': instance.list.map(toJsonT).toList(),
  'total': instance.total,
};
