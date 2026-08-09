import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/user_model.dart';
import '../../providers/auth_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/responsive_utils.dart';
import '../../widgets/logout_confirm_dialog.dart';

/// 我的 — L1 Tab 根页面
///
/// 顶部信息区：用户卡 + VIP 横幅 + 数据统计
/// 分组入口：个人数据 / 商业服务 / 其他 / 管理入口（权限过滤）
class ProfilePage extends ConsumerStatefulWidget {
  const ProfilePage({super.key});

  @override
  ConsumerState<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends ConsumerState<ProfilePage> {
  // 统计缓存
  int? _quotaRemaining;
  int? _favoriteCount;
  int? _taskTotal;
  bool _statsLoaded = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadStats());
  }

  Future<void> _loadStats() async {
    if (_statsLoaded) return;
    _statsLoaded = true;
    // TODO: 接入 MemberService / ModelService / FavoriteService 真实 API
    // 当前使用占位数据展示 UI 结构
    if (!mounted) return;
    setState(() {
      _quotaRemaining = 12;
      _favoriteCount = 5;
      _taskTotal = 3;
    });
  }

  String _displayValue(int? val) => val != null ? '$val' : '-';

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final user = authState.user;
    final theme = Theme.of(context);
    final isWide = ResponsiveUtils.isWideScreen(context);

    if (user == null) {
      return _buildNotLoggedIn(context, theme);
    }

    return Scaffold(
      body: isWide ? _buildWideLayout(context, ref, user, theme) : _buildMobileLayout(context, ref, user, theme),
    );
  }

  // ==================== 移动端布局 ====================

  Widget _buildMobileLayout(BuildContext context, WidgetRef ref, UserModel user, ThemeData theme) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      child: Column(
        children: [
          _buildUserCard(user, theme),
          const SizedBox(height: 12),
          _buildVipBanner(context, theme),
          const SizedBox(height: 12),
          _buildStatsRow(context, theme),
          const SizedBox(height: 20),
          _buildEntryGroup(context, '个人数据', _personalDataEntries, theme),
          const SizedBox(height: 16),
          _buildEntryGroup(context, '商业服务', _businessEntries, theme),
          const SizedBox(height: 16),
          _buildEntryGroup(context, '其他', _otherEntries, theme),
          const SizedBox(height: 16),
          _buildAdminGroups(context, theme),
          const SizedBox(height: 24),
          _buildLogoutButton(context, ref, theme),
          const SizedBox(height: 16),
          Text('图像去雾系统 v1.0', style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
        ],
      ),
    );
  }

  // ==================== 桌面端布局 ====================

  Widget _buildWideLayout(BuildContext context, WidgetRef ref, UserModel user, ThemeData theme) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 900),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildUserCardWide(user, theme),
              const SizedBox(height: 20),
              // 桌面端三列统计卡
              Row(
                children: [
                  Expanded(child: _buildStatCard('剩余额度', _displayValue(_quotaRemaining), Icons.wallet, theme)),
                  const SizedBox(width: 16),
                  Expanded(child: _buildStatCard('处理次数', _displayValue(_taskTotal), Icons.task_alt, theme)),
                  const SizedBox(width: 16),
                  Expanded(child: _buildStatCard('我的收藏', _displayValue(_favoriteCount), Icons.favorite_border, theme)),
                ],
              ),
              const SizedBox(height: 24),
              _buildVipBannerWide(context, theme),
              const SizedBox(height: 24),
              // 桌面端双列入口
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _buildEntryGroup(context, '个人数据', _personalDataEntries, theme)),
                  const SizedBox(width: 16),
                  Expanded(child: _buildEntryGroup(context, '商业服务', _businessEntries, theme)),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _buildEntryGroup(context, '其他', _otherEntries, theme)),
                  const SizedBox(width: 16),
                  const Expanded(child: SizedBox()),
                ],
              ),
              const SizedBox(height: 16),
              _buildAdminGroups(context, theme),
              const SizedBox(height: 24),
              _buildLogoutButton(context, ref, theme),
            ],
          ),
        ),
      ),
    );
  }

  // ==================== 用户卡 ====================

  Widget _buildUserCard(UserModel user, ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: AppTheme.getPrimaryGradient(),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(28),
            ),
            child: Center(
              child: Text(user.avatarInitials, style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w700)),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(user.nickname ?? user.username,
                    style: theme.textTheme.titleLarge?.copyWith(color: Colors.white, fontWeight: FontWeight.w700)),
                const SizedBox(height: 2),
                Text('@${user.username}', style: theme.textTheme.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.8))),
                if (user.roleNames.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Wrap(
                    spacing: 6,
                    children: user.roleNames.take(2).map((role) => Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(role, style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w500)),
                        )).toList(),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUserCardWide(UserModel user, ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        gradient: AppTheme.getPrimaryGradient(),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Row(
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(36),
            ),
            child: Center(
              child: Text(user.avatarInitials, style: const TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.w700)),
            ),
          ),
          const SizedBox(width: 20),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(user.nickname ?? user.username, style: theme.textTheme.headlineSmall?.copyWith(color: Colors.white, fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('@${user.username}', style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white.withValues(alpha: 0.8))),
                if (user.deptName != null)
                  Text(user.deptName!, style: theme.textTheme.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.7))),
              ],
            ),
          ),
          if (user.roleNames.isNotEmpty)
            Wrap(
              spacing: 8,
              children: user.roleNames.take(3).map((role) => Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(role, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
                  )).toList(),
            ),
        ],
      ),
    );
  }

  // ==================== VIP 横幅 ====================

  Widget _buildVipBanner(BuildContext context, ThemeData theme) {
    return InkWell(
      onTap: () => context.push(AppRouterConfig.member),
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          gradient: const LinearGradient(colors: [Color(0xFFFFD700), Color(0xFFFFA500)]),
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
        ),
        child: Row(
          children: [
            const Icon(Icons.workspace_premium, color: Colors.white, size: 22),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('开通 VIP 畅享更多次数', style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white, fontWeight: FontWeight.w600)),
                  Text('解锁全部高级功能', style: theme.textTheme.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.85))),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(8)),
              child: Text('去开通', style: TextStyle(color: AppTheme.brandBlue, fontWeight: FontWeight.w600, fontSize: 13)),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVipBannerWide(BuildContext context, ThemeData theme) {
    return InkWell(
      onTap: () => context.push(AppRouterConfig.member),
      borderRadius: BorderRadius.circular(AppTheme.radiusL),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
        decoration: BoxDecoration(
          gradient: const LinearGradient(colors: [Color(0xFFFFD700), Color(0xFFFFA500)]),
          borderRadius: BorderRadius.circular(AppTheme.radiusL),
        ),
        child: Row(
          children: [
            const Icon(Icons.workspace_premium, color: Colors.white, size: 28),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('开通 VIP 畅享更多次数', style: theme.textTheme.titleMedium?.copyWith(color: Colors.white, fontWeight: FontWeight.w700)),
                  Text('解锁全部高级功能，畅享无限处理', style: theme.textTheme.bodySmall?.copyWith(color: Colors.white.withValues(alpha: 0.85))),
                ],
              ),
            ),
            FilledButton(
              onPressed: () => context.push(AppRouterConfig.member),
              style: FilledButton.styleFrom(backgroundColor: Colors.white, foregroundColor: AppTheme.brandBlue),
              child: const Text('去开通'),
            ),
          ],
        ),
      ),
    );
  }

  // ==================== 数据统计 ====================

  Widget _buildStatsRow(BuildContext context, ThemeData theme) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.dividerColor),
      ),
      child: Row(
        children: [
          _buildStatItem('剩余额度', _displayValue(_quotaRemaining), theme, onTap: () => context.push(AppRouterConfig.quota)),
          Container(width: 1, height: 32, color: theme.dividerColor),
          _buildStatItem('处理次数', _displayValue(_taskTotal), theme, onTap: () => context.push(AppRouterConfig.taskHistory)),
          Container(width: 1, height: 32, color: theme.dividerColor),
          _buildStatItem('我的收藏', _displayValue(_favoriteCount), theme, onTap: () => context.push(AppRouterConfig.favorites)),
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value, ThemeData theme, {VoidCallback? onTap}) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        child: Column(
          children: [
            Text(value, style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(label, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String label, String value, IconData icon, ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
        border: Border.all(color: theme.dividerColor),
      ),
      child: Column(
        children: [
          Icon(icon, color: AppTheme.brandBlue, size: 28),
          const SizedBox(height: 8),
          Text(value, style: theme.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w700, color: AppTheme.brandBlue)),
          const SizedBox(height: 4),
          Text(label, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
        ],
      ),
    );
  }

  // ==================== 入口分组 ====================

  Widget _buildEntryGroup(BuildContext context, String title, List<_EntryItem> entries, ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(title, style: theme.textTheme.labelMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant, fontWeight: FontWeight.w600, letterSpacing: 0.5)),
        ),
        Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            border: Border.all(color: theme.dividerColor),
          ),
          child: Column(
            children: entries.asMap().entries.map((e) {
              final idx = e.key;
              final item = e.value;
              return Column(
                children: [
                  ListTile(
                    leading: Icon(item.icon, color: theme.colorScheme.onSurfaceVariant, size: 22),
                    title: Text(item.title, style: theme.textTheme.bodyMedium),
                    trailing: Icon(Icons.chevron_right, color: theme.colorScheme.onSurfaceVariant, size: 20),
                    onTap: () => context.push(item.route),
                    visualDensity: VisualDensity.compact,
                  ),
                  if (idx < entries.length - 1) Divider(height: 1, indent: 16, endIndent: 16, color: theme.dividerColor),
                ],
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  // ==================== 管理入口（权限过滤） ====================

  Widget _buildAdminGroups(BuildContext context, ThemeData theme) {
    final authState = ref.watch(authProvider);
    final visibleGroups = _adminGroups.where((group) {
      return group.entries.any((entry) => entry.permission == null || authState.hasPerm(entry.permission!));
    }).toList();

    if (visibleGroups.isEmpty) return const SizedBox.shrink();

    return Column(
      children: visibleGroups.map((group) {
        final visibleEntries = group.entries.where((e) => e.permission == null || authState.hasPerm(e.permission!)).toList();
        if (visibleEntries.isEmpty) return const SizedBox.shrink();
        return Padding(
          padding: const EdgeInsets.only(bottom: 16),
          child: _buildEntryGroupRaw(context, group.title, visibleEntries, theme),
        );
      }).toList(),
    );
  }

  Widget _buildEntryGroupRaw(BuildContext context, String title, List<_EntryItem> entries, ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(title, style: theme.textTheme.labelMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant, fontWeight: FontWeight.w600, letterSpacing: 0.5)),
        ),
        Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(AppTheme.radiusL),
            border: Border.all(color: theme.dividerColor),
          ),
          child: Column(
            children: entries.asMap().entries.map((e) {
              final idx = e.key;
              final item = e.value;
              return Column(
                children: [
                  ListTile(
                    leading: Icon(item.icon, color: theme.colorScheme.onSurfaceVariant, size: 22),
                    title: Text(item.title, style: theme.textTheme.bodyMedium),
                    trailing: Icon(Icons.chevron_right, color: theme.colorScheme.onSurfaceVariant, size: 20),
                    onTap: () => context.push(item.route),
                    visualDensity: VisualDensity.compact,
                  ),
                  if (idx < entries.length - 1) Divider(height: 1, indent: 16, endIndent: 16, color: theme.dividerColor),
                ],
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  // ==================== 退出登录 ====================

  Widget _buildLogoutButton(BuildContext context, WidgetRef ref, ThemeData theme) => SizedBox(
        width: double.infinity,
        child: OutlinedButton.icon(
          onPressed: () => showLogoutConfirm(context, ref),
          icon: Icon(Icons.logout, color: theme.colorScheme.error),
          label: Text('退出登录', style: TextStyle(color: theme.colorScheme.error)),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 14),
            side: BorderSide(color: theme.colorScheme.error.withValues(alpha: 0.3)),
          ),
        ),
      );

  // ==================== 未登录 ====================

  Widget _buildNotLoggedIn(BuildContext context, ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.person_off_outlined, size: 64, color: theme.colorScheme.onSurfaceVariant),
            const SizedBox(height: 16),
            Text('请先登录', style: theme.textTheme.titleMedium),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: () => context.go(AppRouterConfig.login),
              child: const Text('去登录'),
            ),
          ],
        ),
      );

  // ==================== 入口数据定义 ====================

  static const _personalDataEntries = [
    _EntryItem(Icons.folder_outlined, '我的文件', AppRouterConfig.files),
    _EntryItem(Icons.dataset_outlined, '我的数据集', AppRouterConfig.dataset),
    _EntryItem(Icons.history, '处理历史', AppRouterConfig.taskHistory),
    _EntryItem(Icons.favorite_border, '我的收藏', AppRouterConfig.favorites),
  ];

  static const _businessEntries = [
    _EntryItem(Icons.workspace_premium_outlined, '我的会员', AppRouterConfig.member),
    _EntryItem(Icons.inventory_2_outlined, '我的套餐', AppRouterConfig.package),
    _EntryItem(Icons.shopping_bag_outlined, '我的订单', AppRouterConfig.orders),
    _EntryItem(Icons.account_balance_wallet_outlined, '我的额度', AppRouterConfig.quota),
    _EntryItem(Icons.feedback_outlined, '反馈评价', AppRouterConfig.feedback),
  ];

  static const _otherEntries = [
    _EntryItem(Icons.settings_outlined, '系统设置', AppRouterConfig.settings),
    _EntryItem(Icons.help_outline, '帮助中心', AppRouterConfig.help),
    _EntryItem(Icons.info_outline, '关于我们', AppRouterConfig.about),
    _EntryItem(Icons.notifications_outlined, '消息设置', AppRouterConfig.notify),
  ];

  // 管理入口分组（权限过滤，无权限整组不显示，路径与 dev-admin 对齐）
  static const _adminGroups = [
    _AdminGroup('工作台', [
      _EntryItem(Icons.dashboard_outlined, '工作台', '/profile/dashboard', permission: 'sys:user:*'),
    ]),
    _AdminGroup('算法与数据', [
      _EntryItem(Icons.science_outlined, '算法管理', '/profile/admin/algorithm-manage', permission: 'sys:algorithm:*'),
      _EntryItem(Icons.storage_outlined, '数据集管理', '/profile/admin/dataset-manage', permission: 'sys:dataset:*'),
    ]),
    _AdminGroup('系统管理', [
      _EntryItem(Icons.people_outline, '用户管理', '/profile/admin/user-manage', permission: 'sys:user:*'),
      _EntryItem(Icons.admin_panel_settings_outlined, '角色管理', '/profile/admin/role-manage', permission: 'sys:role:*'),
      _EntryItem(Icons.menu_open_outlined, '菜单管理', '/profile/admin/menu-manage', permission: 'sys:menu:*'),
      _EntryItem(Icons.business_outlined, '部门管理', '/profile/admin/dept-manage', permission: 'sys:dept:*'),
      _EntryItem(Icons.book_outlined, '字典管理', '/profile/admin/dict-manage', permission: 'sys:dict:*'),
      _EntryItem(Icons.assignment_outlined, '任务管理', '/profile/admin/task-manage', permission: 'sys:task:*'),
    ]),
    _AdminGroup('运营管理', [
      _EntryItem(Icons.card_membership_outlined, '会员管理', '/profile/admin/member-manage', permission: 'sys:member:*'),
      _EntryItem(Icons.inventory_2_outlined, '套餐管理', '/profile/admin/package-manage', permission: 'sys:package:*'),
      _EntryItem(Icons.receipt_long_outlined, '订单管理', '/profile/admin/order-manage', permission: 'sys:order:*'),
      _EntryItem(Icons.rate_review_outlined, '反馈评价管理', '/profile/admin/feedback-manage', permission: 'sys:feedback:*'),
      _EntryItem(Icons.recommend_outlined, '推荐管理', '/profile/admin/recommend-manage', permission: 'sys:recommendation:*'),
      _EntryItem(Icons.campaign_outlined, '消息管理', '/profile/admin/message-manage', permission: 'sys:notify:*'),
    ]),
  ];
}

class _EntryItem {
  final IconData icon;
  final String title;
  final String route;
  final String? permission;

  const _EntryItem(this.icon, this.title, this.route, {this.permission});
}

class _AdminGroup {
  final String title;
  final List<_EntryItem> entries;

  const _AdminGroup(this.title, this.entries);
}
