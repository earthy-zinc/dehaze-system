// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'algorithm_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AlgorithmOption _$AlgorithmOptionFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmOption', json, ($checkedConvert) {
      final val = AlgorithmOption(
        value: $checkedConvert('value', (v) => (v as num).toInt()),
        label: $checkedConvert('label', (v) => v as String),
        type: $checkedConvert('type', (v) => v as String?),
      );
      return val;
    });

Map<String, dynamic> _$AlgorithmOptionToJson(AlgorithmOption instance) =>
    <String, dynamic>{
      'value': instance.value,
      'label': instance.label,
      if (instance.type case final value?) 'type': value,
    };

AlgorithmModel _$AlgorithmModelFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AlgorithmModel', json, ($checkedConvert) {
      final val = AlgorithmModel(
        id: $checkedConvert('id', (v) => (v as num).toInt()),
        name: $checkedConvert('name', (v) => v as String),
        type: $checkedConvert(
          'type',
          (v) => $enumDecode(
            _$AlgorithmTypeEnumMap,
            v,
            unknownValue: AlgorithmType.traditional,
          ),
        ),
        status: $checkedConvert(
          'status',
          (v) =>
              $enumDecodeNullable(_$AlgorithmStatusEnumMap, v) ??
              AlgorithmStatus.disabled,
        ),
        parentId: $checkedConvert('parentId', (v) => (v as num?)?.toInt()),
        description: $checkedConvert('description', (v) => v as String?),
        modelPath: $checkedConvert('modelPath', (v) => v as String?),
        config: $checkedConvert('config', (v) => v as Map<String, dynamic>?),
        remark: $checkedConvert('remark', (v) => v as String?),
        createTime: $checkedConvert('createTime', (v) => v as String?),
        updateTime: $checkedConvert('updateTime', (v) => v as String?),
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
      'type': _$AlgorithmTypeEnumMap[instance.type]!,
      'status': _$AlgorithmStatusEnumMap[instance.status]!,
      if (instance.parentId case final value?) 'parentId': value,
      if (instance.description case final value?) 'description': value,
      if (instance.modelPath case final value?) 'modelPath': value,
      if (instance.config case final value?) 'config': value,
      if (instance.remark case final value?) 'remark': value,
      if (instance.createTime case final value?) 'createTime': value,
      if (instance.updateTime case final value?) 'updateTime': value,
      'children': instance.children.map((e) => e.toJson()).toList(),
    };

const _$AlgorithmTypeEnumMap = {
  AlgorithmType.traditional: 'traditional',
  AlgorithmType.deepLearning: 'deep_learning',
};

const _$AlgorithmStatusEnumMap = {
  AlgorithmStatus.enabled: 1,
  AlgorithmStatus.disabled: 0,
  AlgorithmStatus.auditing: 2,
};
