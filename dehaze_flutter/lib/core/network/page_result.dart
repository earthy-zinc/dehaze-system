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
}
