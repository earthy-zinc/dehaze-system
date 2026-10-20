import 'package:json_annotation/json_annotation.dart';

part 'api_key_model.g.dart';

/// API 密钥视图对象
///
/// 对应 JS SDK ApiKeyVO 与后端 ApiKeyResult。
@JsonSerializable()
class ApiKeyVO {
  const ApiKeyVO({
    required this.id,
    required this.name,
    required this.keyPrefix,
    required this.status,
    this.apiKey,
    this.expiresAt,
    this.lastUsedAt,
    this.createTime,
    this.dailyQuota,
    this.monthlyQuota,
    this.rpmLimit,
    this.modelWhitelist,
  });

  factory ApiKeyVO.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyVOFromJson(json);

  final int id;
  final String name;

  /// 完整密钥，仅创建时返回一次，后续查询不再返回
  final String? apiKey;

  /// 密钥前缀（用于列表展示识别）
  final String keyPrefix;

  /// 状态（1 启用 / 0 禁用）
  final int status;

  final String? expiresAt;
  final String? lastUsedAt;
  final String? createTime;

  /// 日调用配额（NULL 或 0 表示不限制）
  final int? dailyQuota;

  /// 月调用配额（NULL 或 0 表示不限制）
  final int? monthlyQuota;

  /// 每分钟请求数上限（NULL 或 0 表示不限制）
  final int? rpmLimit;

  /// 模型白名单（NULL 或空数组 = 继承用户可见模型）
  final List<String>? modelWhitelist;

  Map<String, dynamic> toJson() => _$ApiKeyVOToJson(this);
}

/// API 密钥创建表单
///
/// 对应 JS SDK ApiKeyCreateForm 与后端 ApiKeyForm。
@JsonSerializable()
class ApiKeyCreateForm {
  const ApiKeyCreateForm({
    required this.name,
    this.expiresAt,
    this.dailyQuota,
    this.monthlyQuota,
    this.rpmLimit,
    this.modelWhitelist,
  });

  factory ApiKeyCreateForm.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyCreateFormFromJson(json);

  final String name;
  final String? expiresAt;
  final int? dailyQuota;
  final int? monthlyQuota;
  final int? rpmLimit;
  final List<String>? modelWhitelist;

  Map<String, dynamic> toJson() => _$ApiKeyCreateFormToJson(this);
}
