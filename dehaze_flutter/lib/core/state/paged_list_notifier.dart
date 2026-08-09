import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../network/page_result.dart';

/// 分页列表快照
class PagedList<T> {
  const PagedList({
    this.items = const [],
    this.total = 0,
    this.pageNum = 1,
    this.pageSize = 10,
  });

  final List<T> items;
  final int total;
  final int pageNum;
  final int pageSize;

  bool get hasMore => items.length < total;
}

/// 分页列表 Notifier 基类
///
/// 统一管理 loading/error/data 状态与首页加载/增量加载逻辑，消除各管理页
/// 重复的 `_items/_total/_pageNum/_loading/_error` 样板。子类仅需实现
/// [fetchPage]，在其中读取 [keyword] 及自定义筛选字段构造查询。
///
/// 子类约定：
/// - 所有筛选字段必须通过字段初始化器赋默认值（不要在构造体中赋值），
///   以保证构造时自动触发的首次加载能读到正确值；
/// - service 通过构造参数注入。
abstract class PagedListNotifier<T> extends StateNotifier<AsyncValue<PagedList<T>>> {
  PagedListNotifier({this.pageSize = 10}) : super(const AsyncValue.loading()) {
    refresh();
  }

  final int pageSize;

  /// 当前搜索关键词，子类在 [fetchPage] 中读取
  @protected
  String keyword = '';

  bool _loadingMore = false;

  /// 子类实现：按页码加载一页数据（读取 [keyword] 及子类自定义筛选字段）
  Future<PageResult<T>> fetchPage(int pageNum);

  /// 设置关键词并重新加载第一页
  Future<void> search(String keyword) async {
    this.keyword = keyword;
    await refresh();
  }

  /// 重新加载第一页（首次加载 / 搜索 / 筛选变更 / 下拉刷新）
  Future<void> refresh() async {
    state = const AsyncValue.loading();
    try {
      final result = await fetchPage(1);
      if (!mounted) return;
      state = AsyncValue.data(
        PagedList<T>(items: result.list, total: result.total, pageNum: 1, pageSize: pageSize),
      );
    } catch (e, st) {
      if (!mounted) return;
      state = AsyncValue.error(e, st);
    }
  }

  /// 增量加载下一页
  Future<void> loadMore() async {
    final current = state.valueOrNull;
    if (current == null || !current.hasMore || _loadingMore) return;
    _loadingMore = true;
    try {
      final result = await fetchPage(current.pageNum + 1);
      if (!mounted) return;
      state = AsyncValue.data(
        PagedList<T>(
          items: [...current.items, ...result.list],
          total: result.total,
          pageNum: current.pageNum + 1,
          pageSize: pageSize,
        ),
      );
    } catch (e, st) {
      if (!mounted) return;
      state = AsyncValue.error(e, st);
    } finally {
      _loadingMore = false;
    }
  }
}

/// 滚动触底加载更多监听器
///
/// 包装可滚动组件，接近底部时触发 [onLoadMore]，替代在 `itemBuilder` 内
/// 直接发起异步请求的写法，避免 build 内副作用与重复触发。
class LoadMoreListener extends StatelessWidget {
  const LoadMoreListener({
    super.key,
    required this.child,
    required this.onLoadMore,
    this.threshold = 200,
  });

  final Widget child;
  final VoidCallback onLoadMore;
  final double threshold;

  @override
  Widget build(BuildContext context) {
    return NotificationListener<ScrollNotification>(
      onNotification: (notification) {
        if (notification is ScrollEndNotification) {
          final metrics = notification.metrics;
          if (metrics.pixels >= metrics.maxScrollExtent - threshold) {
            onLoadMore();
          }
        }
        return false;
      },
      child: child,
    );
  }
}
