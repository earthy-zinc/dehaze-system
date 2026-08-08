import 'package:json_annotation/json_annotation.dart';

part 'dict_model.g.dart';

// ==================== DictType ====================

/// 字典类型
@JsonSerializable()
class DictType {
  const DictType({
    required this.id,
    required this.name,
    required this.code,
    this.status,
    this.remark,
    this.createTime,
    this.updateTime,
  });

  factory DictType.fromJson(Map<String, dynamic> json) =>
      _$DictTypeFromJson(json);

  final int id;
  final String name;
  final String code;
  final int? status;
  final String? remark;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DictTypeToJson(this);
}

// ==================== DictTypeQuery ====================

/// 字典类型查询参数
@JsonSerializable()
class DictTypeQuery {
  const DictTypeQuery({
    this.pageNum,
    this.pageSize,
    this.keywords,
  });

  factory DictTypeQuery.fromJson(Map<String, dynamic> json) =>
      _$DictTypeQueryFromJson(json);

  final int? pageNum;
  final int? pageSize;
  final String? keywords;

  Map<String, dynamic> toJson() => _$DictTypeQueryToJson(this);

  Map<String, dynamic> toQuery() => {
        if (pageNum != null) 'pageNum': pageNum,
        if (pageSize != null) 'pageSize': pageSize,
        if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
      };
}

// ==================== DictTypeForm ====================

/// 字典类型表单
@JsonSerializable()
class DictTypeForm {
  const DictTypeForm({
    this.id,
    this.name,
    this.code,
    required this.status,
    this.remark,
  });

  factory DictTypeForm.fromJson(Map<String, dynamic> json) =>
      _$DictTypeFormFromJson(json);

  final int? id;
  final String? name;
  final String? code;
  final int status;
  final String? remark;

  Map<String, dynamic> toJson() => _$DictTypeFormToJson(this);
}

// ==================== Dict ====================

/// 字典项
@JsonSerializable()
class Dict {
  const Dict({
    this.id,
    this.typeId,
    this.typeCode,
    this.label,
    this.value,
    this.sort,
    this.status,
    this.remark,
    this.cssClass,
    this.listClass,
    this.isDefault,
    this.createTime,
    this.updateTime,
  });

  factory Dict.fromJson(Map<String, dynamic> json) => _$DictFromJson(json);

  final int? id;

  @JsonKey(name: 'typeId')
  final int? typeId;

  @JsonKey(name: 'typeCode')
  final String? typeCode;

  final String? label;
  final String? value;
  final int? sort;
  final int? status;
  final String? remark;

  @JsonKey(name: 'cssClass')
  final String? cssClass;

  @JsonKey(name: 'listClass')
  final String? listClass;

  @JsonKey(name: 'isDefault')
  final int? isDefault;

  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DictToJson(this);
}

// ==================== DictQuery ====================

/// 字典项查询参数
@JsonSerializable()
class DictQuery {
  const DictQuery({
    this.pageNum,
    this.pageSize,
    this.typeCode,
    this.keywords,
  });

  factory DictQuery.fromJson(Map<String, dynamic> json) =>
      _$DictQueryFromJson(json);

  final int? pageNum;
  final int? pageSize;

  @JsonKey(name: 'typeCode')
  final String? typeCode;

  final String? keywords;

  Map<String, dynamic> toJson() => _$DictQueryToJson(this);

  Map<String, dynamic> toQuery() => {
        if (pageNum != null) 'pageNum': pageNum,
        if (pageSize != null) 'pageSize': pageSize,
        if (typeCode != null && typeCode!.isNotEmpty) 'typeCode': typeCode,
        if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
      };
}

// ==================== DictForm ====================

/// 字典项表单
@JsonSerializable()
class DictForm {
  const DictForm({
    this.id,
    this.typeId,
    this.typeCode,
    this.label,
    this.value,
    this.sort,
    this.status,
    this.remark,
    this.cssClass,
    this.listClass,
    this.isDefault,
  });

  factory DictForm.fromJson(Map<String, dynamic> json) =>
      _$DictFormFromJson(json);

  final int? id;

  @JsonKey(name: 'typeId')
  final int? typeId;

  @JsonKey(name: 'typeCode')
  final String? typeCode;

  final String? label;
  final String? value;
  final int? sort;
  final int? status;
  final String? remark;

  @JsonKey(name: 'cssClass')
  final String? cssClass;

  @JsonKey(name: 'listClass')
  final String? listClass;

  @JsonKey(name: 'isDefault')
  final int? isDefault;

  Map<String, dynamic> toJson() => _$DictFormToJson(this);
}

// ==================== DictOption ====================

/// 字典选项（下拉框用）
@JsonSerializable()
class DictOption {
  const DictOption({
    required this.value,
    required this.label,
    this.children,
  });

  factory DictOption.fromJson(Map<String, dynamic> json) =>
      _$DictOptionFromJson(json);

  final String value;
  final String label;
  final List<DictOption>? children;

  Map<String, dynamic> toJson() => _$DictOptionToJson(this);
}
