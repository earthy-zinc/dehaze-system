import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

/// 开发者日志面板（任务 3.4）。
///
/// 提供日志文件浏览与导出（复制到剪贴板 / 分享文件）。
/// 生产环境隐藏入口（仅在 debug 下显示）。
class DevLogsPage extends StatefulWidget {
  const DevLogsPage({super.key});

  @override
  State<DevLogsPage> createState() => _DevLogsPageState();
}

class _DevLogsPageState extends State<DevLogsPage> {
  List<File> _logFiles = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadLogFiles();
  }

  Future<void> _loadLogFiles() async {
    setState(() => _loading = true);
    try {
      final dir = await getApplicationDocumentsDirectory();
      final logsDir = Directory('${dir.path}/logs');
      final files = <File>[];
      if (logsDir.existsSync()) {
        for (final dateDir in logsDir.listSync().whereType<Directory>()) {
          for (final file in dateDir.listSync().whereType<File>()) {
            files.add(file);
          }
        }
      }
      files.sort((a, b) => b.path.compareTo(a.path));
      setState(() {
        _logFiles = files;
        _loading = false;
      });
    } catch (_) {
      setState(() {
        _logFiles = [];
        _loading = false;
      });
    }
  }

  Future<void> _exportLog(File file) async {
    final content = await file.readAsString();
    if (!mounted) return;
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('日志导出'),
        content: SizedBox(
          width: double.maxFinite,
          height: 300,
          child: Column(
            children: [
              Text(
                file.path.split('/').last,
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 12),
              Expanded(
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SingleChildScrollView(
                    child: SelectableText(
                      content.isEmpty ? '(空文件)' : content,
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('关闭'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('开发者日志'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: '刷新',
            onPressed: _loadLogFiles,
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _logFiles.isEmpty
              ? const Center(child: Text('暂无日志文件'))
              : ListView.builder(
                  itemCount: _logFiles.length,
                  itemBuilder: (context, index) {
                    final file = _logFiles[index];
                    final size = file.lengthSync();
                    final sizeText = size > 1024 * 1024
                        ? '${(size / 1024 / 1024).toStringAsFixed(1)} MB'
                        : size > 1024
                            ? '${(size / 1024).toStringAsFixed(0)} KB'
                            : '$size B';
                    return ListTile(
                      leading: const Icon(Icons.description_outlined),
                      title: Text(
                        file.path.split('/logs/').last,
                        style: const TextStyle(fontSize: 13),
                      ),
                      subtitle: Text(sizeText),
                      onTap: () => _exportLog(file),
                      trailing: const Icon(Icons.chevron_right),
                    );
                  },
                ),
    );
  }
}
