// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'algorithm_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AlgorithmModel _$AlgorithmModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmModel', json, ($checkedConvert) {
      final val = AlgorithmModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String? ?? '未分类'),
        status: $checkedConvert(
          'status',
          (v) =>
              $enumDecodeNullable(_$AlgorithmStatusEnumMap, v) ??
              AlgorithmStatus.draft,
        ),
        parentId: $checkedConvert('parentId', (v) => (v as num?)?.toInt()),
        description: $checkedConvert('description', (v) => v as String?),
        path: $checkedConvert('path', (v) => v as String?),
        importPath: $checkedConvert('importPath', (v) => v as String?),
        children: $checkedConvert(
          'children',
          (v) =>
              (v as List<dynamic>?)
                  ?.map(
                    (e) => AlgorithmModel.fromJson(e as Map<String, dynamic>),
                  )
                  .toList() ??
              const [],
        ),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmModelToJson(AlgorithmModel instance) =>
    <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'type': instance.type,
      'status': _$AlgorithmStatusEnumMap[instance.status]!,
      if (instance.parentId case final value?) 'parentId': value,
      if (instance.description case final value?) 'description': value,
      if (instance.path case final value?) 'path': value,
      if (instance.importPath case final value?) 'importPath': value,
      'children': instance.children.map((e) => e.toJson()).toList(),
    };

const _$AlgorithmStatusEnumMap = {
  AlgorithmStatus.draft: 0,
  AlgorithmStatus.testing: 1,
  AlgorithmStatus.pendingAudit: 2,
  AlgorithmStatus.published: 3,
  AlgorithmStatus.disabled: 4,
  AlgorithmStatus.archived: 5,
};
