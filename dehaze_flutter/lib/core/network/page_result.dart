import 'package:json_annotation/json_annotation.dart';

part 'page_result.g.dart';

/// 分页响应模型
///
/// 对应后端 PageResult 结构：
/// ```json
/// {
///   "list": [],
///   "total": 100
/// }
/// ```
@JsonSerializable(genericArgumentFactories: true)
class PageResult<T> {
  const PageResult({
    required this.list,
    required this.total,
  });

  factory PageResult.fromJson(
    Map<String, dynamic> json,
    T Function(Object? json) fromJsonT,
  ) =>
      _$PageResultFromJson(json, fromJsonT);

  /// 当前页数据列表
  @JsonKey(defaultValue: [])
  final List<T> list;

  /// 总记录数
  @JsonKey(defaultValue: 0)
  final int total;

  Map<String, dynamic> toJson(Object? Function(T value) toJsonT) =>
      _$PageResultToJson(this, toJsonT);
}
