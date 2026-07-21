import 'package:json_annotation/json_annotation.dart';

part 'algorithm_model.g.dart';

/// 算法状态枚举
enum AlgorithmStatus {
  @JsonValue(1)
  enabled,
  @JsonValue(0)
  disabled,
  @JsonValue(2)
  auditing,
}

extension AlgorithmStatusExtension on AlgorithmStatus {
  String get displayName {
    switch (this) {
      case AlgorithmStatus.enabled:
        return '已启用';
      case AlgorithmStatus.disabled:
        return '已禁用';
      case AlgorithmStatus.auditing:
        return '审核中';
    }
  }
}

/// 算法选项（下拉选择用）
@JsonSerializable()
class AlgorithmOption {
  const AlgorithmOption({
    required this.value,
    required this.label,
    this.type,
  });

  factory AlgorithmOption.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmOptionFromJson(json);

  final int value;
  final String label;
  final String? type;

  Map<String, dynamic> toJson() => _$AlgorithmOptionToJson(this);
}

/// 算法信息
@JsonSerializable()
class AlgorithmModel {
  const AlgorithmModel({
    required this.id,
    required this.name,
    required this.type,
    required this.status,
    this.parentId,
    this.description,
    this.path,
    this.importPath,
    this.config,
    this.remark,
    this.createTime,
    this.updateTime,
    this.children = const [],
  });

  factory AlgorithmModel.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmModelFromJson(json);

  final int id;
  final String name;

  /// 算法任务分类（后端返回中文，如"图像去雾"、"图像去噪"、"图像去雨"、"图像去模糊"）
  @JsonKey(defaultValue: '未分类')
  final String type;

  @JsonKey(defaultValue: AlgorithmStatus.disabled)
  final AlgorithmStatus status;

  @JsonKey(name: 'parentId')
  final int? parentId;

  final String? description;

  /// 模型文件相对路径（如 AECR-Net/NH_train.pk）
  final String? path;

  /// 模型导入路径（如 algorithm.AECRNet.run，深度学习模型才有）
  final String? importPath;

  /// 算法配置参数（JSON 字符串）
  final Map<String, dynamic>? config;

  final String? remark;

  @JsonKey(name: 'createTime')
  final String? createTime;

  @JsonKey(name: 'updateTime')
  final String? updateTime;

  /// 子算法列表（树形结构）
  final List<AlgorithmModel> children;

  Map<String, dynamic> toJson() => _$AlgorithmModelToJson(this);

  /// 是否为深度学习算法（后端以 importPath 标识可导入的模型）
  bool get isDeepLearning =>
      importPath != null && importPath!.trim().isNotEmpty;

  /// 是否已启用
  bool get isEnabled => status == AlgorithmStatus.enabled;
}
