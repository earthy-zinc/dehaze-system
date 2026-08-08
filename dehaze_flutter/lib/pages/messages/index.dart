import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/message_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/ui_utils.dart';

class MessagesPage extends ConsumerStatefulWidget {
  const MessagesPage({super.key});

  @override
  ConsumerState<MessagesPage> createState() => _MessagesPageState();
}

class _MessagesPageState extends ConsumerState<MessagesPage> {
  String? _activeType;
  int _unreadCount = 0;
  List<MessageVO> _messages = [];
  bool _loading = true;

  static const _typeTabs = [
    _TypeTab(label: '全部', value: null),
    _TypeTab(label: '系统', value: 'system'),
    _TypeTab(label: '处理', value: 'task'),
    _TypeTab(label: '活动', value: 'activity'),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadData();
    });
  }

  Future<void> _loadData() async {
    setState(() => _loading = true);
    try {
      final service = ref.read(messageServiceProvider);
      final unread = await service.getUnreadCount();
      final pageData = await service.getPage(
        type: _activeType,
        pageNum: 1,
        pageSize: 50,
      );
      final list = (pageData['data']?['list'] as List<dynamic>?)
          ?.map((e) => MessageVO.fromJson(e as Map<String, dynamic>))
          .toList() ?? [];

      if (mounted) {
        setState(() {
          _unreadCount = unread.count;
          _messages = list;
          _loading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _loading = false);
        showError(context, '加载消息失败');
      }
    }
  }

  Future<void> _markRead(int id) async {
    try {
      await ref.read(messageServiceProvider).markRead(id);
      _loadData();
    } catch (e) {
      if (mounted) showError(context, '操作失败');
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      body: Column(
        children: [
          _buildHeader(context, theme),
          _buildTypeTabs(theme),
          Expanded(child: _buildMessageList(theme)),
        ],
      ),
    );
  }

  Widget _buildHeader(BuildContext context, ThemeData theme) {
    return SafeArea(
      bottom: false,
      child: Padding(
        padding: EdgeInsets.fromLTRB(
          AppTheme.spacingM,
          AppTheme.spacingS,
          AppTheme.spacingS,
          AppTheme.spacingS,
        ),
        child: Row(
          children: [
            Text(
              '消息中心',
              style: theme.textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
            if (_unreadCount > 0) ...[
              SizedBox(width: AppTheme.spacingS),
              Container(
                padding: EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: AppTheme.errorColor,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(
                  '$_unreadCount',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
            Spacer(),
            IconButton(
              icon: Icon(Icons.settings_outlined, size: 22),
              tooltip: '消息设置',
              onPressed: () => context.go(AppRouterConfig.notify),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTypeTabs(ThemeData theme) {
    return Container(
      height: 44,
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(color: theme.dividerColor),
        ),
      ),
      child: Row(
        children: _typeTabs.map((tab) {
          final isActive = tab.value == _activeType;
          return Expanded(
            child: GestureDetector(
              onTap: () {
                setState(() => _activeType = tab.value);
                _loadData();
              },
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    tab.label,
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: isActive
                          ? AppTheme.brandBlue
                          : theme.colorScheme.onSurfaceVariant,
                      fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                  if (isActive)
                    Container(
                      margin: EdgeInsets.only(top: 8),
                      width: 24,
                      height: 3,
                      decoration: BoxDecoration(
                        color: AppTheme.brandBlue,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                ],
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildMessageList(ThemeData theme) {
    if (_loading) {
      return Center(child: CircularProgressIndicator());
    }

    if (_messages.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.mail_outline, size: 64, color: theme.colorScheme.onSurfaceVariant),
            SizedBox(height: AppTheme.spacingM),
            Text(
              '暂无消息',
              style: theme.textTheme.bodyLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadData,
      child: ListView.separated(
        padding: EdgeInsets.symmetric(vertical: AppTheme.spacingS),
        itemCount: _messages.length + 1,
        separatorBuilder: (_, _) => Divider(height: 1, indent: AppTheme.spacingXL),
        itemBuilder: (ctx, index) {
          if (index == _messages.length) {
            return SizedBox(height: 80);
          }
          return _MessageItem(
            message: _messages[index],
            onTap: () async {
              final msg = _messages[index];
              final goRouter = GoRouter.of(context);
              await _markRead(msg.id);
              if (!mounted) return;
              goRouter.go('/messages/${msg.id}');
            },
          );
        },
      ),
    );
  }
}

class _TypeTab {
  const _TypeTab({required this.label, required this.value});
  final String label;
  final String? value;
}

class _MessageItem extends StatelessWidget {
  const _MessageItem({required this.message, required this.onTap});

  final MessageVO message;
  final VoidCallback onTap;

  IconData _typeIcon(String type) {
    switch (type) {
      case 'system':
        return Icons.campaign_outlined;
      case 'task':
        return Icons.auto_fix_high;
      case 'activity':
        return Icons.celebration_outlined;
      default:
        return Icons.notifications_outlined;
    }
  }

  Color _typeColor(String type) {
    switch (type) {
      case 'system':
        return AppTheme.brandBlue;
      case 'task':
        return AppTheme.techGreen;
      case 'activity':
        return AppTheme.warningColor;
      default:
        return AppTheme.gray500;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = _typeColor(message.type);

    return ListTile(
      onTap: onTap,
      contentPadding: EdgeInsets.symmetric(
        horizontal: AppTheme.spacingM,
        vertical: AppTheme.spacingXS,
      ),
      leading: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(AppTheme.radiusM),
        ),
        child: Icon(_typeIcon(message.type), color: color, size: 22),
      ),
      title: Row(
        children: [
          Expanded(
            child: Text(
              message.title,
              style: theme.textTheme.bodyMedium?.copyWith(
                fontWeight: message.isRead ? FontWeight.w400 : FontWeight.w600,
              ),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          if (!message.isRead)
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: AppTheme.errorColor,
                shape: BoxShape.circle,
              ),
            ),
        ],
      ),
      subtitle: Text(
        message.summary ?? '',
        style: theme.textTheme.bodySmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: Text(
        _formatTime(message.createTime),
        style: theme.textTheme.labelSmall?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }

  String _formatTime(String time) {
    try {
      final dt = DateTime.parse(time);
      final now = DateTime.now();
      final diff = now.difference(dt);
      if (diff.inMinutes < 1) return '刚刚';
      if (diff.inHours < 1) return '${diff.inMinutes}分钟前';
      if (diff.inDays < 1) return '${diff.inHours}小时前';
      if (diff.inDays < 7) return '${diff.inDays}天前';
      return '${dt.month}/${dt.day}';
    } catch (_) {
      return '';
    }
  }
}
