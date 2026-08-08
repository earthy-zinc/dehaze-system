import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/file_model.dart';
import '../../providers/providers.dart';

/// 我的文件 — L2 页面
///
/// 对接 FileService.getPage 真实 API。
class FilesPage extends ConsumerStatefulWidget {
  const FilesPage({super.key});

  @override
  ConsumerState<FilesPage> createState() => _FilesPageState();
}

class _FilesPageState extends ConsumerState<FilesPage> {
  final List<FileInfo> _items = [];
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
      final service = ref.read(fileServiceProvider);
      final result = await service.getPage(
        const FileQuery(pageNum: 1, pageSize: _pageSize),
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
      final service = ref.read(fileServiceProvider);
      final nextPage = _pageNum + 1;
      final result = await service.getPage(
        FileQuery(pageNum: nextPage, pageSize: _pageSize),
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

  IconData _fileIcon(String? type) {
    if (type == null) return Icons.insert_drive_file_outlined;
    switch (type.toLowerCase()) {
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'webp':
      case 'bmp':
      case 'svg':
        return Icons.image_outlined;
      case 'mp4':
      case 'avi':
      case 'mov':
      case 'mkv':
      case 'wmv':
        return Icons.video_file_outlined;
      case 'pdf':
        return Icons.picture_as_pdf_outlined;
      case 'zip':
      case 'rar':
      case '7z':
      case 'tar':
      case 'gz':
        return Icons.folder_zip_outlined;
      default:
        return Icons.insert_drive_file_outlined;
    }
  }

  String _formatTime(String? time) {
    if (time == null || time.isEmpty) return '';
    if (time.length >= 16) return time.substring(0, 16).replaceFirst('T', ' ');
    return time;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的文件')),
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
                            return _buildFileCard(item, theme);
                          },
                        ),
                      ),
                    ),
    );
  }

  bool get _hasMore => _items.length < _total;

  Widget _buildFileCard(FileInfo item, ThemeData theme) {
    final isImage = item.type != null &&
        ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'svg']
            .contains(item.type!.toLowerCase());

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: isImage
            ? ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: Image.network(
                  item.url,
                  width: 48,
                  height: 48,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) =>
                      Icon(_fileIcon(item.type), size: 36),
                ),
              )
            : Icon(_fileIcon(item.type), size: 36),
        title: Text(
          item.name,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (item.size != null && item.size!.isNotEmpty)
              Text(
                '${item.size}  ${item.type ?? ""}',
                style: theme.textTheme.bodySmall,
              ),
            if (_formatTime(item.createTime).isNotEmpty)
              Text(
                _formatTime(item.createTime),
                style: theme.textTheme.labelSmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
          ],
        ),
        trailing: const Icon(Icons.chevron_right),
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
            Icon(Icons.folder_open,
                size: 64,
                color: theme.colorScheme.onSurface.withValues(alpha: 0.3)),
            const SizedBox(height: 16),
            Text('暂无文件',
                style: theme.textTheme.titleMedium
                    ?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      );
}
