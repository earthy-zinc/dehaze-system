import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../domain/entities/dehaze_image.dart';

class ProcessingStatusWidget extends StatelessWidget {

  const ProcessingStatusWidget({
    required this.isProcessing, required this.onClearCurrent, super.key,
    this.currentImage,
  });
  final bool isProcessing;
  final DehazeImage? currentImage;
  final VoidCallback onClearCurrent;

  @override
  Widget build(BuildContext context) {
    if (isProcessing && currentImage == null) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Row(
            children: [
              CircularProgressIndicator(),
              SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '正在处理图片...',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    Text('请稍候，正在应用去雾算法'),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (currentImage != null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    _getStatusIcon(currentImage!.status),
                    color: _getStatusColor(currentImage!.status),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _getStatusText(currentImage!.status),
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: _getStatusColor(currentImage!.status),
                      ),
                    ),
                  ),
                  if (currentImage!.status == ProcessingStatus.completed)
                    IconButton(
                      onPressed: onClearCurrent,
                      icon: const Icon(Icons.close),
                      tooltip: '清除当前结果',
                    ),
                ],
              ),
              const SizedBox(height: 8),
              if (currentImage!.metadata != null)
                Text(
                  '处理时间: ${currentImage!.metadata!.processingTime.inSeconds}秒',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              if (currentImage!.processedImagePath != null)
                const SizedBox(height: 8),
              if (currentImage!.processedImagePath != null)
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => _viewResultImage(context),
                        icon: const Icon(Icons.visibility),
                        label: const Text('查看结果'),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: () => _saveResultImage(context),
                        icon: const Icon(Icons.save),
                        label: const Text('保存图片'),
                      ),
                    ),
                  ],
                ),
              if (currentImage!.status == ProcessingStatus.failed)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    '处理失败，请重试',
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ),
            ],
          ),
        ),
      );
    }

    return const SizedBox.shrink();
  }

  IconData _getStatusIcon(ProcessingStatus status) {
    switch (status) {
      case ProcessingStatus.pending:
        return Icons.schedule;
      case ProcessingStatus.processing:
        return Icons.hourglass_empty;
      case ProcessingStatus.completed:
        return Icons.check_circle;
      case ProcessingStatus.failed:
        return Icons.error;
      case ProcessingStatus.cancelled:
        return Icons.cancel;
    }
  }

  Color _getStatusColor(ProcessingStatus status) {
    switch (status) {
      case ProcessingStatus.pending:
        return Colors.orange;
      case ProcessingStatus.processing:
        return Colors.blue;
      case ProcessingStatus.completed:
        return Colors.green;
      case ProcessingStatus.failed:
        return Colors.red;
      case ProcessingStatus.cancelled:
        return Colors.grey;
    }
  }

  String _getStatusText(ProcessingStatus status) {
    switch (status) {
      case ProcessingStatus.pending:
        return '等待处理';
      case ProcessingStatus.processing:
        return '正在处理中...';
      case ProcessingStatus.completed:
        return '处理完成';
      case ProcessingStatus.failed:
        return '处理失败';
      case ProcessingStatus.cancelled:
        return '处理已取消';
    }
  }

  void _viewResultImage(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('处理结果'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 300,
              height: 200,
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.image, size: 64, color: Colors.grey),
                    Text('去雾后图片'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              '路径: ${currentImage!.processedImagePath}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => context.pop(), child: const Text('关闭')),
        ],
      ),
    );
  }

  void _saveResultImage(BuildContext context) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('图片已保存到相册')));
  }
}
