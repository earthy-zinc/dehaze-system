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
    this.img,
    this.path,
    this.importPath,
    this.params,
    this.flops,
    this.size,
    this.version,
    this.auditBy,
    this.auditTime,
    this.auditRemark,
    this.createTime,
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

  final int? parentId;

  final String? description;

  /// 算法图片 URL
  final String? img;

  /// 模型文件相对路径（如 AECR-Net/NH_train.pk）
  final String? path;

  /// 模型导入路径（如 algorithm.AECRNet.run，深度学习模型才有）
  final String? importPath;

  /// 参数量
  final String? params;

  /// FLOPs（浮点运算次数）
  final String? flops;

  /// 模型文件大小
  final String? size;

  /// 算法版本
  final String? version;

  /// 审核人 ID
  final int? auditBy;

  /// 审核时间
  final String? auditTime;

  /// 审核备注
  final String? auditRemark;

  /// 创建时间
  final String? createTime;

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

// ============================================================================
// 以下为算法模块扩展模型（对齐 JS SDK algorithm/model.ts）
// ============================================================================

/// 算法查询参数
class AlgorithmQuery {
  const AlgorithmQuery({this.keywords});

  final String? keywords;

  Map<String, dynamic> toQuery() => {
        if (keywords != null) 'keywords': keywords,
      };
}

/// 算法审核表单
@JsonSerializable()
class AlgorithmAuditForm {
  const AlgorithmAuditForm({
    required this.approved,
    this.remark,
  });

  factory AlgorithmAuditForm.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmAuditFormFromJson(json);

  final bool approved;
  final String? remark;

  Map<String, dynamic> toJson() => _$AlgorithmAuditFormToJson(this);
}

/// 算法版本创建表单
@JsonSerializable()
class AlgorithmVersionForm {
  const AlgorithmVersionForm({
    required this.version,
    this.changeLog,
    this.modelFileId,
  });

  factory AlgorithmVersionForm.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmVersionFormFromJson(json);

  final String version;
  final String? changeLog;

  @JsonKey(name: 'modelFileId')
  final int? modelFileId;

  Map<String, dynamic> toJson() => _$AlgorithmVersionFormToJson(this);
}

/// 算法版本 VO
@JsonSerializable()
class AlgorithmVersionVO {
  const AlgorithmVersionVO({
    required this.id,
    required this.algorithmId,
    required this.version,
    this.changeLog,
    this.status,
    this.isActive,
    this.modelFileId,
    this.createTime,
  });

  factory AlgorithmVersionVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmVersionVOFromJson(json);

  final int id;

  @JsonKey(name: 'algorithmId')
  final int algorithmId;

  final String version;
  final String? changeLog;
  final int? status;

  @JsonKey(name: 'isActive')
  final bool? isActive;

  @JsonKey(name: 'modelFileId')
  final int? modelFileId;

  final String? createTime;

  Map<String, dynamic> toJson() => _$AlgorithmVersionVOToJson(this);
}

/// 算法监控数据
@JsonSerializable()
class AlgorithmMonitorVO {
  const AlgorithmMonitorVO({
    required this.callCount,
    required this.avgTime,
    required this.successRate,
    required this.todayCallCount,
  });

  factory AlgorithmMonitorVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmMonitorVOFromJson(json);

  final int callCount;
  final double avgTime;
  final double successRate;
  final int todayCallCount;

  Map<String, dynamic> toJson() => _$AlgorithmMonitorVOToJson(this);
}

/// 算法监控统计条目
@JsonSerializable()
class AlgorithmMonitorStatsItemVO {
  const AlgorithmMonitorStatsItemVO({
    required this.date,
    required this.callCount,
    required this.avgTime,
    required this.successRate,
  });

  factory AlgorithmMonitorStatsItemVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmMonitorStatsItemVOFromJson(json);

  final String date;
  final int callCount;
  final double avgTime;
  final double successRate;

  Map<String, dynamic> toJson() => _$AlgorithmMonitorStatsItemVOToJson(this);
}

/// 算法对比表单
class AlgorithmCompareForm {
  const AlgorithmCompareForm({
    required this.algorithmIds,
    this.fileId,
    this.imageUrl,
  });

  final List<int> algorithmIds;
  final int? fileId;
  final String? imageUrl;

  Map<String, dynamic> toJson() => {
        'algorithmIds': algorithmIds,
        if (fileId != null) 'fileId': fileId,
        if (imageUrl != null) 'imageUrl': imageUrl,
      };
}

/// 算法对比结果项
@JsonSerializable()
class AlgorithmCompareVO {
  const AlgorithmCompareVO({
    required this.algorithmId,
    required this.algorithmName,
    this.resultUrl,
    this.time,
    this.metrics,
  });

  factory AlgorithmCompareVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmCompareVOFromJson(json);

  final int algorithmId;
  final String algorithmName;
  final String? resultUrl;
  final int? time;

  /// 评估指标（PSNR/SSIM 等，JSON 字符串）
  final String? metrics;

  Map<String, dynamic> toJson() => _$AlgorithmCompareVOToJson(this);
}

/// 算法选择树节点
@JsonSerializable()
class AlgorithmSelectNodeVO {
  const AlgorithmSelectNodeVO({
    required this.id,
    required this.parentId,
    required this.name,
    required this.type,
    required this.leaf,
    this.children,
  });

  factory AlgorithmSelectNodeVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmSelectNodeVOFromJson(json);

  final int id;
  final int parentId;
  final String name;
  final String type;
  final bool leaf;
  final List<AlgorithmSelectNodeVO>? children;

  Map<String, dynamic> toJson() => _$AlgorithmSelectNodeVOToJson(this);
}

/// 算法详情 VO（用户端 select/{id}）
@JsonSerializable()
class AlgorithmDetailVO {
  const AlgorithmDetailVO({
    required this.id,
    required this.name,
    required this.type,
    required this.description,
    this.img,
    this.path,
    this.size,
    this.params,
    this.flops,
    this.version,
    this.status,
    this.avgRating,
    this.ratingCount,
    this.usageCount,
    this.sampleImages,
  });

  factory AlgorithmDetailVO.fromJson(Map<String, dynamic> json) =>
      _$AlgorithmDetailVOFromJson(json);

  final int id;
  final String name;
  final String type;
  final String? img;
  final String description;
  final String? path;
  final String? size;
  final String? params;
  final String? flops;
  final String? version;
  final int? status;
  final double? avgRating;
  final int? ratingCount;
  final int? usageCount;
  final List<String>? sampleImages;

  Map<String, dynamic> toJson() => _$AlgorithmDetailVOToJson(this);
}

/// 算法测试表单
class AlgorithmTestForm {
  const AlgorithmTestForm({this.fileId, this.imageUrl, this.params});

  final int? fileId;
  final String? imageUrl;

  /// 预测参数（JSON 字符串）
  final String? params;

  Map<String, dynamic> toJson() => {
        if (fileId != null) 'fileId': fileId,
        if (imageUrl != null) 'imageUrl': imageUrl,
        if (params != null) 'params': params,
      };
}
