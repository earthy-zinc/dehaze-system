import 'package:json_annotation/json_annotation.dart';

part 'algorithm_model.g.dart';

/// 算法类型枚举
enum AlgorithmType {
  @JsonValue('traditional')
  traditional,
  @JsonValue('deep_learning')
  deepLearning,
}

extension AlgorithmTypeExtension on AlgorithmType {
  String get displayName {
    switch (this) {
      case AlgorithmType.traditional:
        return '传统算法';
      case AlgorithmType.deepLearning:
        return '深度学习';
    }
  }
}

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
    this.modelPath,
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

  @JsonKey(unknownEnumValue: AlgorithmType.traditional)
  final AlgorithmType type;

  @JsonKey(defaultValue: AlgorithmStatus.disabled)
  final AlgorithmStatus status;

  @JsonKey(name: 'parentId')
  final int? parentId;

  final String? description;

  @JsonKey(name: 'modelPath')
  final String? modelPath;

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

  /// 是否为深度学习算法
  bool get isDeepLearning => type == AlgorithmType.deepLearning;

  /// 是否已启用
  bool get isEnabled => status == AlgorithmStatus.enabled;
}
