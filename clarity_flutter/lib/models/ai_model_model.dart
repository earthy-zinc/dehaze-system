import 'package:json_annotation/json_annotation.dart';

part 'ai_model_model.g.dart';

/// AI 模型视图对象（对齐后端 AiModelVO / SDK AiModelVO，仅保留前端消费字段）
@JsonSerializable()
class AiModelVO {
  const AiModelVO({
    required this.modelId,
    required this.displayName,
    this.modelType,
    this.taskTags,
  });

  factory AiModelVO.fromJson(Map<String, dynamic> json) =>
      _$AiModelVOFromJson(json);

  /// 模型业务标识（统一模型规范 slug；提交图像任务时作为 `model`）
  final String modelId;

  /// 显示名称
  final String displayName;

  /// 模型类型（image_edit 等）
  final String? modelType;

  /// 任务归类标签（图像模型用于选择场景过滤，如 ["dehaze"]）
  final List<String>? taskTags;

  Map<String, dynamic> toJson() => _$AiModelVOToJson(this);
}
