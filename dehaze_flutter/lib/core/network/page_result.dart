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

/// 分页请求参数
class PageRequest {
  const PageRequest({
    this.pageNum = 1,
    this.pageSize = 10,
    this.keywords,
  });

  /// 页码（从 1 开始）
  final int pageNum;

  /// 每页条数
  final int pageSize;

  /// 搜索关键词
  final String? keywords;

  /// 转换为查询参数
  Map<String, dynamic> toQueryParams() => {
        'pageNum': pageNum,
        'pageSize': pageSize,
        if (keywords != null && keywords!.isNotEmpty) 'keywords': keywords,
      };

  /// 下一页请求
  PageRequest nextPage() => PageRequest(
        pageNum: pageNum + 1,
        pageSize: pageSize,
        keywords: keywords,
      );
}
