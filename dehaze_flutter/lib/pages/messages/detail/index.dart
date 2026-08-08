import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../models/message_model.dart';
import '../../../providers/providers.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/ui_utils.dart';

class MessageDetailPage extends ConsumerStatefulWidget {
  const MessageDetailPage({required this.messageId, super.key});

  final int messageId;

  @override
  ConsumerState<MessageDetailPage> createState() => _MessageDetailPageState();
}

class _MessageDetailPageState extends ConsumerState<MessageDetailPage> {
  MessageVO? _message;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadDetail());
  }

  Future<void> _loadDetail() async {
    setState(() => _loading = true);
    try {
      final message = await ref
          .read(messageServiceProvider)
          .getDetail(widget.messageId);
      if (mounted) {
        setState(() {
          _message = message;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _loading = false);
        showError(context, '加载消息详情失败');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: Text('消息详情')),
      body: _loading
          ? Center(child: CircularProgressIndicator())
          : _message == null
              ? Center(child: Text('消息不存在'))
              : SingleChildScrollView(
                  padding: EdgeInsets.all(AppTheme.spacingM),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _message!.title,
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      SizedBox(height: AppTheme.spacingS),
                      Row(
                        children: [
                          Container(
                            padding: EdgeInsets.symmetric(
                              horizontal: AppTheme.spacingS,
                              vertical: AppTheme.spacingXS,
                            ),
                            decoration: BoxDecoration(
                              color: AppTheme.brandBlue.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(AppTheme.radiusS),
                            ),
                            child: Text(
                              _message!.typeLabel,
                              style: theme.textTheme.labelSmall?.copyWith(
                                color: AppTheme.brandBlue,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                          SizedBox(width: AppTheme.spacingS),
                          Text(
                            _message!.createTime,
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: AppTheme.spacingL),
                      Divider(),
                      SizedBox(height: AppTheme.spacingL),
                      Text(
                        _message!.content ?? _message!.summary ?? '',
                        style: theme.textTheme.bodyLarge?.copyWith(
                          height: 1.7,
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }
}
