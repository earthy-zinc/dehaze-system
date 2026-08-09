import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../providers/auth_provider.dart';
import '../../theme/app_theme.dart';

/// 工作台页面（L2 管理入口）
///
/// 统计卡片 + 管理功能快捷入口，仅管理员可见
class DashboardPage extends ConsumerWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authState = ref.watch(authProvider);
    final theme = Theme.of(context);
    final isWide = MediaQuery.of(context).size.width >= 768;

    return Scaffold(
      appBar: AppBar(title: const Text('工作台')),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(AppTheme.spacingM),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildWelcomeCard(theme, authState.user?.nickname ?? authState.user?.username ?? '管理员'),
            SizedBox(height: AppTheme.spacingL),
            Text('快捷入口', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
            SizedBox(height: AppTheme.spacingM),
            _buildQuickGrid(context, theme, isWide, authState),
          ],
        ),
      ),
    );
  }

  Widget _buildWelcomeCard(ThemeData theme, String name) => Container(
    width: double.infinity,
    padding: EdgeInsets.all(AppTheme.spacingL),
    decoration: BoxDecoration(
      gradient: AppTheme.getPrimaryGradient(),
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('欢迎回来，$name', style: theme.textTheme.titleLarge?.copyWith(color: Colors.white, fontWeight: FontWeight.w700)),
        SizedBox(height: AppTheme.spacingS),
        Text('Dehaze 管理系统工作台', style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white.withValues(alpha: 0.8))),
      ],
    ),
  );

  Widget _buildQuickGrid(BuildContext context, ThemeData theme, bool isWide, AuthState auth) {
    final entries = _getQuickEntries(auth);
    final crossAxisCount = isWide ? 4 : 2;

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        mainAxisSpacing: AppTheme.spacingM,
        crossAxisSpacing: AppTheme.spacingM,
        childAspectRatio: 1.3,
      ),
      itemCount: entries.length,
      itemBuilder: (context, index) {
        final entry = entries[index];
        return Card(
          child: InkWell(
            onTap: () => context.go(entry.route),
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            child: Padding(
              padding: EdgeInsets.all(AppTheme.spacingM),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(entry.icon, size: 32, color: AppTheme.brandBlue),
                  SizedBox(height: AppTheme.spacingS),
                  Text(entry.title, style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w600), textAlign: TextAlign.center),
                  SizedBox(height: 2),
                  Text(entry.desc, style: theme.textTheme.bodySmall, textAlign: TextAlign.center, maxLines: 2, overflow: TextOverflow.ellipsis),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  List<_QuickEntry> _getQuickEntries(AuthState auth) {
    final entries = <_QuickEntry>[];
    if (auth.hasPerm('sys:user:*')) entries.add(_QuickEntry(Icons.people, '用户管理', '系统用户增删改查', '/profile/admin/user-manage'));
    if (auth.hasPerm('sys:algorithm:*')) entries.add(_QuickEntry(Icons.science, '算法管理', '算法审核与上下架', '/profile/admin/algorithm-manage'));
    if (auth.hasPerm('sys:dataset:*')) entries.add(_QuickEntry(Icons.storage, '数据集管理', '数据集创建与维护', '/profile/admin/dataset-manage'));
    if (auth.hasPerm('sys:task:*')) entries.add(_QuickEntry(Icons.task, '任务管理', '全用户任务监控', '/profile/admin/task-manage'));
    if (auth.hasPerm('sys:order:*')) entries.add(_QuickEntry(Icons.receipt_long, '订单管理', '订单处理与退款', '/profile/admin/order-manage'));
    if (auth.hasPerm('sys:member:*')) entries.add(_QuickEntry(Icons.card_membership, '会员管理', '会员等级与权益', '/profile/admin/member-manage'));
    if (auth.hasPerm('sys:package:*')) entries.add(_QuickEntry(Icons.shopping_bag, '套餐管理', '套餐配置与上下架', '/profile/admin/package-manage'));
    if (auth.hasPerm('sys:notify:*')) entries.add(_QuickEntry(Icons.campaign, '消息管理', '公告/模板/群发', '/profile/admin/message-manage'));
    return entries;
  }
}

class _QuickEntry {
  final IconData icon;
  final String title;
  final String desc;
  final String route;
  const _QuickEntry(this.icon, this.title, this.desc, this.route);
}
