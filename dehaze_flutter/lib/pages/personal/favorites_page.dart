import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/favorite_model.dart';
import '../../providers/providers.dart';

/// 我的收藏 — L2 页面
///
/// 对接 FavoriteService.getPage / remove 真实 API。
class FavoritesPage extends ConsumerStatefulWidget {
  const FavoritesPage({super.key});

  @override
  ConsumerState<FavoritesPage> createState() => _FavoritesPageState();
}

class _FavoritesPageState extends ConsumerState<FavoritesPage> {
  final List<FavoriteVO> _items = [];
  bool _isLoading = true;
  String? _error;
  int _pageNum = 1;
  int _total = 0;
  bool _isLoadingMore = false;
  static const int _pageSize = 20;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
      _pageNum = 1;
    });
    try {
      final service = ref.read(favoriteServiceProvider);
      final result = await service.getPage(
        const FavoriteQuery(
          targetType: FavoriteTargetType.algorithm,
          pageNum: 1,
          pageSize: _pageSize,
        ),
      );
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _items
          ..clear()
          ..addAll(result.list);
        _total = result.total;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _loadMore() async {
    if (_isLoadingMore || _items.length >= _total) return;
    setState(() => _isLoadingMore = true);
    try {
      final service = ref.read(favoriteServiceProvider);
      final nextPage = _pageNum + 1;
      final result = await service.getPage(
        FavoriteQuery(
          targetType: FavoriteTargetType.algorithm,
          pageNum: nextPage,
          pageSize: _pageSize,
        ),
      );
      if (!mounted) return;
      setState(() {
        _isLoadingMore = false;
        _items.addAll(result.list);
        _pageNum = nextPage;
        _total = result.total;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _isLoadingMore = false);
    }
  }

  Future<void> _removeFavorite(int index) async {
    final item = _items[index];
    try {
      await ref.read(favoriteServiceProvider).remove(
            item.targetId,
            item.targetType,
          );
      if (!mounted) return;
      setState(() => _items.removeAt(index));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('取消收藏失败: $e')),
      );
    }
  }

  String _formatTime(String time) {
    // createTime 可能为 "2025-01-15T10:30:00" 格式，截取前16位展示
    if (time.length >= 16) return time.substring(0, 16).replaceFirst('T', ' ');
    return time;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的收藏')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _buildError(theme)
              : _items.isEmpty
                  ? _buildEmpty(theme)
                  : RefreshIndicator(
                      onRefresh: _load,
                      child: NotificationListener<ScrollNotification>(
                        onNotification: (notification) {
                          if (notification is ScrollEndNotification &&
                              notification.metrics.pixels >=
                                  notification.metrics.maxScrollExtent - 100) {
                            _loadMore();
                          }
                          return false;
                        },
                        child: ListView.builder(
                          padding: const EdgeInsets.all(16),
                          itemCount: _items.length + (_hasMore ? 1 : 0),
                          itemBuilder: (context, index) {
                            if (index >= _items.length) {
                              return const Padding(
                                padding: EdgeInsets.symmetric(vertical: 16),
                                child:
                                    Center(child: CircularProgressIndicator()),
                              );
                            }
                            final item = _items[index];
                            return _buildItemCard(item, index, theme);
                          },
                        ),
                      ),
                    ),
    );
  }

  bool get _hasMore => _items.length < _total;

  Widget _buildItemCard(FavoriteVO item, int index, ThemeData theme) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: item.targetImage != null && item.targetImage!.isNotEmpty
            ? ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: Image.network(
                  item.targetImage!,
                  width: 48,
                  height: 48,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) =>
                      const Icon(Icons.favorite, color: Colors.red),
                ),
              )
            : const Icon(Icons.favorite, color: Colors.red),
        title: Text(item.targetName ?? '未命名'),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (item.targetDescription != null &&
                item.targetDescription!.isNotEmpty)
              Text(
                item.targetDescription!,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: theme.textTheme.bodySmall,
              ),
            const SizedBox(height: 2),
            Text(
              _formatTime(item.createTime),
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
        trailing: IconButton(
          icon: const Icon(Icons.close, size: 18),
          onPressed: () => _removeFavorite(index),
        ),
      ),
    );
  }

  Widget _buildError(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
            const SizedBox(height: 12),
            Text(_error!, style: theme.textTheme.bodyMedium),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _load, child: const Text('重试')),
          ],
        ),
      );

  Widget _buildEmpty(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.favorite_border,
                size: 64,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
            const SizedBox(height: 16),
            Text('暂无收藏',
                style: theme.textTheme.titleMedium
                    ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );
}
