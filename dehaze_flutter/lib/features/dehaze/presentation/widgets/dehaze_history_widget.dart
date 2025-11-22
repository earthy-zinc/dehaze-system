import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../domain/entities/dehaze_image.dart';

class DehazeHistoryWidget extends StatelessWidget {

  const DehazeHistoryWidget({
    required this.history, required this.onDeleteImage, super.key,
  });
  final List<DehazeImage> history;
  final void Function(String) onDeleteImage;

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.history, size: 48, color: Colors.grey[400]),
                const SizedBox(height: 12),
                Text(
                  '暂无处理历史',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey[600],
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  '开始处理图片后，历史记录将显示在这里',
                  style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      );
    }

    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                const Icon(Icons.history),
                const SizedBox(width: 8),
                Text('处理历史', style: Theme.of(context).textTheme.titleLarge),
                const Spacer(),
                Text(
                  '${history.length} 个项目',
                  style: Theme.of(
                    context,
                  ).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          SizedBox(
            height: 300, // 限制历史记录的最大高度
            child: ListView.builder(
              itemCount: history.length,
              itemBuilder: (context, index) {
                final image = history[index];
                return DehazeHistoryItem(
                  image: image,
                  onDelete: () => _showDeleteConfirmation(context, image),
                  onTap: () => _showImageDetails(context, image),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  void _showDeleteConfirmation(BuildContext context, DehazeImage image) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除确认'),
        content: Text('确定要删除这个去雾结果吗？\n\nID: ${image.id}'),
        actions: [
          TextButton(onPressed: () => context.pop(), child: const Text('取消')),
          ElevatedButton(
            onPressed: () {
              onDeleteImage(image.id);
              context.pop();
            },
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: const Text('删除'),
          ),
        ],
      ),
    );
  }

  void _showImageDetails(BuildContext context, DehazeImage image) {
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('图片详情'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: double.infinity,
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
                      Text('原始图片'),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  color: Colors.grey[100],
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
              _buildDetailRow('图片ID', image.id),
              _buildDetailRow('创建时间', _formatDateTime(image.createdAt)),
              if (image.processedAt != null)
                _buildDetailRow('处理完成时间', _formatDateTime(image.processedAt!)),
              _buildDetailRow('状态', _getStatusText(image.status)),
              _buildDetailRow(
                '算法',
                _getAlgorithmName(image.parameters.algorithm),
              ),
              _buildDetailRow(
                '强度',
                '${(image.parameters.strength * 100).toInt()}%',
              ),
              _buildDetailRow(
                '对比度',
                image.parameters.contrast.toStringAsFixed(1),
              ),
              _buildDetailRow(
                '亮度',
                image.parameters.brightness.toStringAsFixed(1),
              ),
              if (image.metadata != null) ...[
                _buildDetailRow(
                  '处理时间',
                  '${image.metadata!.processingTime.inSeconds}秒',
                ),
                _buildDetailRow(
                  '原始大小',
                  '${(image.metadata!.originalSize / 1024).toStringAsFixed(1)}KB',
                ),
                _buildDetailRow(
                  '处理后大小',
                  '${(image.metadata!.processedSize / 1024).toStringAsFixed(1)}KB',
                ),
                _buildDetailRow(
                  '压缩比',
                  '${(image.metadata!.compressionRatio * 100).toInt()}%',
                ),
              ],
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => context.pop(), child: const Text('关闭')),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) => Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );

  String _formatDateTime(DateTime dateTime) => '${dateTime.year}-${dateTime.month.toString().padLeft(2, '0')}-${dateTime.day.toString().padLeft(2, '0')} ${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';

  String _getStatusText(ProcessingStatus status) {
    switch (status) {
      case ProcessingStatus.pending:
        return '等待处理';
      case ProcessingStatus.processing:
        return '处理中';
      case ProcessingStatus.completed:
        return '已完成';
      case ProcessingStatus.failed:
        return '失败';
      case ProcessingStatus.cancelled:
        return '已取消';
    }
  }

  String _getAlgorithmName(DehazeAlgorithm algorithm) {
    switch (algorithm) {
      case DehazeAlgorithm.darkChannel:
        return '暗通道先验';
      case DehazeAlgorithm.atmosphericLight:
        return '大气光估计';
      case DehazeAlgorithm.retinex:
        return 'Retinex理论';
      case DehazeAlgorithm.colorAttenuation:
        return '颜色衰减先验';
      case DehazeAlgorithm.custom:
        return '自定义算法';
    }
  }
}

class DehazeHistoryItem extends StatelessWidget {

  const DehazeHistoryItem({
    required this.image, required this.onDelete, required this.onTap, super.key,
  });
  final DehazeImage image;
  final VoidCallback onDelete;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: ListTile(
        dense: true,
        leading: CircleAvatar(
          radius: 16,
          backgroundColor: _getStatusColor(image.status),
          child: Icon(
            _getStatusIcon(image.status),
            color: Colors.white,
            size: 16,
          ),
        ),
        title: Text(
          '去雾图片 #${image.id.substring(0, 8)}',
          style: const TextStyle(fontWeight: FontWeight.w500, fontSize: 14),
        ),
        subtitle: Text(
          '${_getAlgorithmName(image.parameters.algorithm)} • ${_formatDateTime(image.createdAt)}',
          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
        ),
        trailing: IconButton(
          onPressed: onDelete,
          icon: const Icon(Icons.delete_outline, size: 20),
          tooltip: '删除',
          visualDensity: VisualDensity.compact,
        ),
        onTap: onTap,
      ),
    );

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

  String _getAlgorithmName(DehazeAlgorithm algorithm) {
    switch (algorithm) {
      case DehazeAlgorithm.darkChannel:
        return '暗通道先验';
      case DehazeAlgorithm.atmosphericLight:
        return '大气光估计';
      case DehazeAlgorithm.retinex:
        return 'Retinex理论';
      case DehazeAlgorithm.colorAttenuation:
        return '颜色衰减先验';
      case DehazeAlgorithm.custom:
        return '自定义算法';
    }
  }

  String _formatDateTime(DateTime dateTime) => '${dateTime.month}-${dateTime.day} ${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
}
