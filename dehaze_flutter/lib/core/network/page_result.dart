/// 分页响应模型
class PageResult<T> {
  const PageResult({
    required this.list,
    required this.total,
  });

  /// 当前页数据列表
  final List<T> list;

  /// 总记录数
  final int total;

  /// 从后端分页响应体（`data` 字段，含 list/total）构造 PageResult
  ///
  /// 消除各 service 重复的 `data['list'] / data['total']` 解包样板。
  factory PageResult.fromResponse(
    Map<String, dynamic> data,
    T Function(Map<String, dynamic> json) fromJson,
  ) {
    return PageResult<T>(
      list: (data['list'] as List<dynamic>)
          .map((e) => fromJson(e as Map<String, dynamic>))
          .toList(),
      total: (data['total'] as num).toInt(),
    );
  }
}
