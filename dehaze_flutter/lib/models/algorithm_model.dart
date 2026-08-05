import 'package:json_annotation/json_annotation.dart';

part 'algorithm_model.g.dart';

enum AlgorithmStatus {
  @JsonValue(0)
  draft,
  @JsonValue(1)
  testing,
  @JsonValue(2)
  pendingAudit,
  @JsonValue(3)
  published,
  @JsonValue(4)
  disabled,
  @JsonValue(5)
  archived,
}

extension AlgorithmStatusExtension on AlgorithmStatus {
  String get displayName {
    switch (this) {
      case AlgorithmStatus.draft:
        return '草稿';
      case AlgorithmStatus.testing:
        return '测试中';
      case AlgorithmStatus.pendingAudit:
        return '待审核';
      case AlgorithmStatus.published:
        return '已发布';
      case AlgorithmStatus.disabled:
        return '已停用';
      case AlgorithmStatus.archived:
        return '已归档';
    }
  }
}

/// 算法信息（对应后端 AlgorithmVO）
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
    this.children = const [],
  });

  factory AlgorithmModel.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmModelFromJson(json);

  final int id;
  final String name;

  /// 算法任务分类（后端返回中文，如"图像去雾"、"图像去噪"、"图像去雨"、"图像去模糊"）
  @JsonKey(defaultValue: '未分类')
  final String type;

  @JsonKey(defaultValue: AlgorithmStatus.draft)
  final AlgorithmStatus status;

  @JsonKey(name: 'parentId')
  final int? parentId;

  final String? description;

  /// 模型文件相对路径（如 AECR-Net/NH_train.pk）
  final String? path;

  /// 模型导入路径（如 algorithm.AECRNet.run，深度学习模型才有）
  @JsonKey(name: 'importPath')
  final String? importPath;

  /// 子算法列表（树形结构）
  final List<AlgorithmModel> children;

  Map<String, dynamic> toJson() => _$AlgorithmModelToJson(this);

  /// 是否为深度学习算法（后端以 importPath 标识可导入的模型）
  bool get isDeepLearning =>
      importPath != null && importPath!.trim().isNotEmpty;
}

/// 算法列表展平工具
extension AlgorithmListExtension on List<AlgorithmModel> {
  /// 展平树形结构，返回所有已发布（status==published）的叶子算法。
  ///
  /// 约定：父节点为分类节点（不含可执行模型），叶子节点为具体算法。
  /// 移动端仅允许选择已发布的算法，供算法选择/算法信息页使用。
  List<AlgorithmModel> get flatPublishedLeaves {
    final result = <AlgorithmModel>[];
    void collect(List<AlgorithmModel> nodes) {
      for (final node in nodes) {
        if (node.children.isEmpty) {
          if (node.status == AlgorithmStatus.published) result.add(node);
        } else {
          collect(node.children);
        }
      }
    }

    collect(this);
    return result;
  }
}
