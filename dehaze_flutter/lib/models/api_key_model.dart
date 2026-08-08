import 'package:json_annotation/json_annotation.dart';

part 'api_key_model.g.dart';

/// API 密钥状态枚举
///
/// 对应后端 ApiKey status 字段。
enum ApiKeyStatus {
  /// 活跃（正常可用）
  @JsonValue(1)
  active,

  /// 禁用
  @JsonValue(0)
  inactive,

  /// 已过期
  @JsonValue(2)
  expired,

  /// 已吊销
  @JsonValue(3)
  revoked;

  int get value => switch (this) {
        ApiKeyStatus.active => 1,
        ApiKeyStatus.inactive => 0,
        ApiKeyStatus.expired => 2,
        ApiKeyStatus.revoked => 3,
      };

  static ApiKeyStatus fromValue(int? value) {
    switch (value) {
      case 0:
        return ApiKeyStatus.inactive;
      case 2:
        return ApiKeyStatus.expired;
      case 3:
        return ApiKeyStatus.revoked;
      default:
        return ApiKeyStatus.active;
    }
  }
}

/// API 密钥视图对象
///
/// 对应 JS SDK ApiKeyVO 及后端 ApiKeyResult。
@JsonSerializable()
class ApiKeyVO {
  const ApiKeyVO({
    required this.id,
    this.userId,
    this.name,
    this.keyPrefix,
    this.apiKey,
    this.permissions = const [],
    required this.status,
    this.statusName,
    this.lastUsedTime,
    this.expireTime,
    this.createTime,
  });

  factory ApiKeyVO.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyVOFromJson(json);

  final int id;
  final int? userId;
  final String? name;

  @JsonKey(name: 'keyPrefix')
  final String? keyPrefix;

  @JsonKey(name: 'apiKey')
  final String? apiKey;

  @JsonKey(defaultValue: [])
  final List<String> permissions;

  final int status;
  final String? statusName;

  @JsonKey(name: 'lastUsedAt')
  final String? lastUsedTime;

  @JsonKey(name: 'expiresAt')
  final String? expireTime;

  final String? createTime;

  Map<String, dynamic> toJson() => _$ApiKeyVOToJson(this);
}

/// API 密钥查询参数
@JsonSerializable()
class ApiKeyQuery {
  const ApiKeyQuery({
    this.pageNum = 1,
    this.pageSize = 10,
    this.status,
    this.keyword,
  });

  factory ApiKeyQuery.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyQueryFromJson(json);

  final int pageNum;
  final int pageSize;
  final int? status;
  final String? keyword;

  Map<String, dynamic> toJson() => _$ApiKeyQueryToJson(this);

  /// 转为 queryParameters 格式
  Map<String, dynamic> toQuery() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (status != null) 'status': status,
        if (keyword != null) 'keyword': keyword,
      };
}

/// API 密钥创建表单
///
/// 对应 JS SDK ApiKeyCreateForm 及后端 ApiKeyForm。
@JsonSerializable()
class ApiKeyCreateForm {
  const ApiKeyCreateForm({
    required this.name,
    this.permissions = const [],
    this.expireTime,
  });

  factory ApiKeyCreateForm.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyCreateFormFromJson(json);

  final String name;

  @JsonKey(defaultValue: [])
  final List<String> permissions;

  @JsonKey(name: 'expiresAt')
  final String? expireTime;

  Map<String, dynamic> toJson() => _$ApiKeyCreateFormToJson(this);
}

/// API 密钥更新表单
@JsonSerializable()
class ApiKeyUpdateForm {
  const ApiKeyUpdateForm({
    this.name,
    this.permissions,
    this.status,
  });

  factory ApiKeyUpdateForm.fromJson(Map<String, dynamic> json) =>
      _$ApiKeyUpdateFormFromJson(json);

  final String? name;
  final List<String>? permissions;
  final int? status;

  Map<String, dynamic> toJson() => _$ApiKeyUpdateFormToJson(this);
}
