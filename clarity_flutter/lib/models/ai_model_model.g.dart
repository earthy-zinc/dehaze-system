// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'ai_model_model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

AiModelVO _$AiModelVOFromJson(Map<String, dynamic> json) =>
    $checkedCreate('AiModelVO', json, ($checkedConvert) {
      final val = AiModelVO(
        modelId: $checkedConvert('modelId', (v) => v as String),
        displayName: $checkedConvert('displayName', (v) => v as String),
        modelType: $checkedConvert('modelType', (v) => v as String?),
        taskTags: $checkedConvert(
          'taskTags',
          (v) => (v as List<dynamic>?)?.map((e) => e as String).toList(),
        ),
      );
      return val;
    });

Map<String, dynamic> _$AiModelVOToJson(AiModelVO instance) => <String, dynamic>{
  'modelId': instance.modelId,
  'displayName': instance.displayName,
  'modelType': ?instance.modelType,
  'taskTags': ?instance.taskTags,
};
