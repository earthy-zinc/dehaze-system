import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// 帮助中心 — L2 页面
///
/// FAQ 静态列表
class HelpPage extends ConsumerWidget {
  const HelpPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('帮助中心')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('常见问题', style: theme.textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 16),
          ..._faqs.map((faq) => Card(
                margin: const EdgeInsets.only(bottom: 8),
                child: ExpansionTile(
                  leading: const Icon(Icons.help_outline, size: 22),
                  title: Text(faq['q'] as String, style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600)),
                  children: [
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                      child: Text(faq['a'] as String, style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                    ),
                  ],
                ),
              )),
          const SizedBox(height: 24),
          Text('联系我们', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text('如有其他问题，请通过「我的 > 反馈评价」提交反馈，或发送邮件至 support@dehaze.com', style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
        ],
      ),
    );
  }

  static const _faqs = [
    {'q': '如何开始使用去雾功能？', 'a': '点击底部「去雾」Tab，上传图片后选择算法即可开始处理。您也可以从首页或工具页面快速进入。'},
    {'q': '支持哪些图像格式？', 'a': '支持 JPG、PNG、BMP、TIFF 等常见格式，单张图片大小不超过 20MB。'},
    {'q': '处理需要多长时间？', 'a': '单张图片通常需要 5-30 秒，具体取决于图片大小和所选算法。VIP 用户享有优先处理。'},
    {'q': '如何查看处理历史？', 'a': '在「我的」页面中点击「处理历史」即可查看所有处理记录，支持结果对比和重新处理。'},
    {'q': 'VIP 有哪些权益？', 'a': 'VIP 用户享有更多处理次数、高清输出、优先处理、批量处理等权益。详见「我的 > 我的会员」。'},
    {'q': '如何联系客服？', 'a': '通过「我的 > 反馈评价」提交反馈，我们的团队会尽快回复。'},
  ];
}
