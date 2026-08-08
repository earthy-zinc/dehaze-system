import 'package:json_annotation/json_annotation.dart';

part 'task_model.g.dart';

// ==================== 枚举 ====================

/// 任务状态（对齐后端 TaskConstants: PENDING=1, PROCESSING=2, COMPLETED=3, FAILED=4, CANCELLED=5）
///
/// 注意：prediction_model.dart 中已有 TaskStatus 枚举（仅3个值：processing/completed/failed），
/// 语义和值域均不同，因此此处独立定义 TaskStatusType 避免冲突。
enum TaskStatusType {
  @JsonValue(1)
  pending,
  @JsonValue(2)
  processing,
  @JsonValue(3)
  completed,
  @JsonValue(4)
  failed,
  @JsonValue(5)
  cancelled;

  static TaskStatusType fromValue(int? value) {
    switch (value) {
      case 1:
        return TaskStatusType.pending;
      case 2:
        return TaskStatusType.processing;
      case 3:
        return TaskStatusType.completed;
      case 4:
        return TaskStatusType.failed;
      case 5:
        return TaskStatusType.cancelled;
      default:
        return TaskStatusType.pending;
    }
  }
}

/// 任务类别（对齐后端 TaskConstants.CATEGORY_IMPORT/CATEGORY_EXPORT）
enum TaskCategory {
  @JsonValue('import')
  import_,
  @JsonValue('export')
  export_;

  static TaskCategory fromValue(String? value) {
    switch (value) {
      case 'import':
        return TaskCategory.import_;
      case 'export':
        return TaskCategory.export_;
      default:
        return TaskCategory.export_;
    }
  }
}

// ==================== JSON 转换辅助函数 ====================

TaskStatusType _statusFromJson(dynamic value) {
  if (value is int) return TaskStatusType.fromValue(value);
  if (value is String) return TaskStatusType.fromValue(int.tryParse(value));
  return TaskStatusType.pending;
}

int? _statusToJson(TaskStatusType? status) => status?.index == null ? null : (status!.index + 1);

TaskCategory _categoryFromJson(dynamic value) {
  if (value is String) return TaskCategory.fromValue(value);
  return TaskCategory.export_;
}

String? _categoryToJson(TaskCategory? category) => category?.name.endsWith('_') == true
    ? category!.name.substring(0, category.name.length - 1)
    : category?.name;

// ==================== 模型 ====================

/// 任务 VO（对齐后端 TaskVO）
@JsonSerializable()
class TaskVO {
  const TaskVO({
    required this.taskId,
    required this.status,
    required this.progress,
    this.taskType,
    this.taskCategory,
    this.totalFiles,
    this.processedFiles,
    this.downloadUrl,
    this.expiresAt,
    this.createdAt,
    this.startedAt,
    this.completedAt,
    this.error,
  });

  factory TaskVO.fromJson(Map<String, dynamic> json) => _$TaskVOFromJson(json);

  /// 任务ID
  final String taskId;

  /// 任务状态
  @JsonKey(fromJson: _statusFromJson, toJson: _statusToJson)
  final TaskStatusType status;

  /// 进度（0-100）
  final int progress;

  /// 任务类型：dataset_export, user_export, user_import 等
  final String? taskType;

  /// 任务类别：import / export
  @JsonKey(fromJson: _categoryFromJson, toJson: _categoryToJson)
  final TaskCategory? taskCategory;

  /// 文件总数
  final int? totalFiles;

  /// 已处理文件数
  final int? processedFiles;

  /// 下载链接（任务完成时返回）
  final String? downloadUrl;

  /// 过期时间
  final String? expiresAt;

  /// 创建时间
  final String? createdAt;

  /// 开始执行时间
  final String? startedAt;

  /// 完成时间
  final String? completedAt;

  /// 错误信息（失败时返回）
  final String? error;

  Map<String, dynamic> toJson() => _$TaskVOToJson(this);
}

/// 任务创建表单（对齐 JS SDK TaskCreateForm）
///
/// 注意：JS SDK 将除 type 外的字段打包为 paramsJson 发送。
/// 此处保持与 JS SDK 一致的结构。
@JsonSerializable()
class TaskCreateForm {
  const TaskCreateForm({
    required this.type,
    this.targetId,
    this.targetIds,
    this.options,
  });

  factory TaskCreateForm.fromJson(Map<String, dynamic> json) =>
      _$TaskCreateFormFromJson(json);

  /// 任务类型（如 dataset_export）
  final String type;

  /// 目标资源ID（导出单个资源时使用）
  final int? targetId;

  /// 目标资源ID列表（批量导出时使用）
  final List<int>? targetIds;

  /// 导出选项配置（文件组织方式、包含类型等）
  final Map<String, dynamic>? options;

  Map<String, dynamic> toJson() => _$TaskCreateFormToJson(this);
}

/// 任务查询参数（对齐 JS SDK TaskQuery）
class TaskQuery {
  const TaskQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.status,
    this.taskType,
    this.taskCategory,
  });

  /// 页码
  final int pageNum;

  /// 每页条数
  final int pageSize;

  /// 任务状态筛选
  final TaskStatusType? status;

  /// 任务类型筛选（支持逗号分隔多个）
  final String? taskType;

  /// 任务类别筛选：import / export
  final TaskCategory? taskCategory;

  Map<String, dynamic> toQueryParameters() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (status != null) 'status': _statusToJson(status),
        if (taskType != null && taskType!.isNotEmpty) 'taskType': taskType,
        if (taskCategory != null) 'taskCategory': _categoryToJson(taskCategory),
      };
}
