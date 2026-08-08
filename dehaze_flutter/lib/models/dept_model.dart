import 'package:json_annotation/json_annotation.dart';

part 'dept_model.g.dart';

// ==================== Dept ====================

@JsonSerializable()
class Dept {
  const Dept({
    required this.id,
    required this.parentId,
    required this.name,
    required this.sort,
    required this.status,
    this.leader,
    this.phone,
    this.email,
    this.children,
    this.createTime,
    this.updateTime,
  });

  factory Dept.fromJson(Map<String, dynamic> json) => _$DeptFromJson(json);

  final int id;

  @JsonKey(name: 'parentId')
  final int parentId;

  final String name;
  final int sort;
  final int status;
  final String? leader;
  final String? phone;
  final String? email;
  final List<Dept>? children;
  final String? createTime;
  final String? updateTime;

  Map<String, dynamic> toJson() => _$DeptToJson(this);
}

// ==================== DeptQuery ====================

@JsonSerializable()
class DeptQuery {
  const DeptQuery({this.name, this.status});

  factory DeptQuery.fromJson(Map<String, dynamic> json) =>
      _$DeptQueryFromJson(json);

  final String? name;
  final int? status;

  Map<String, dynamic> toJson() => _$DeptQueryToJson(this);

  Map<String, dynamic> toQuery() => {
        if (name != null) 'keywords': name,
        if (status != null) 'status': status,
      };
}

// ==================== DeptForm ====================

@JsonSerializable()
class DeptForm {
  const DeptForm({
    this.id,
    required this.parentId,
    required this.name,
    required this.sort,
    required this.status,
    this.leader,
    this.phone,
    this.email,
  });

  factory DeptForm.fromJson(Map<String, dynamic> json) =>
      _$DeptFormFromJson(json);

  final int? id;

  @JsonKey(name: 'parentId')
  final int parentId;

  final String name;
  final int sort;
  final int status;
  final String? leader;
  final String? phone;
  final String? email;

  Map<String, dynamic> toJson() => _$DeptFormToJson(this);
}

// ==================== DeptOption ====================

@JsonSerializable()
class DeptOption {
  const DeptOption({
    required this.id,
    required this.name,
    this.children,
  });

  factory DeptOption.fromJson(Map<String, dynamic> json) =>
      _$DeptOptionFromJson(json);

  final int id;
  final String name;
  final List<DeptOption>? children;

  Map<String, dynamic> toJson() => _$DeptOptionToJson(this);
}
