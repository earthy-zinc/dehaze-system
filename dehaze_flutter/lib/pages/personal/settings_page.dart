import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/notification_settings_model.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../utils/ui_utils.dart';
import '../../widgets/logout_confirm_dialog.dart';

/// 系统设置 — L2 页面
///
/// 通知设置（从 API 加载）+ 账号安全、缓存清理、关于、退出登录
class SettingsPage extends ConsumerStatefulWidget {
  const SettingsPage({super.key});

  @override
  ConsumerState<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends ConsumerState<SettingsPage> {
  NotificationSettings? _settings;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadSettings());
  }

  Future<void> _loadSettings() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final service = ref.read(notificationSettingsServiceProvider);
      final settings = await service.getSettings();
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _settings = settings;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  Future<void> _updateSetting(NotificationSettingsForm Function(NotificationSettings s) buildForm) async {
    final s = _settings;
    if (s == null) return;
    final form = buildForm(s);
    // 乐观更新 UI
    setState(() {
      _settings = NotificationSettings(
        id: s.id,
        userId: s.userId,
        systemEnabled: form.systemEnabled,
        predictionEnabled: form.predictionEnabled,
        activityEnabled: form.activityEnabled,
        announcementEnabled: form.announcementEnabled,
        emailEnabled: form.emailEnabled,
        smsEnabled: form.smsEnabled,
        pushEnabled: form.pushEnabled,
        digestEnabled: form.digestEnabled,
        digestFrequency: form.digestFrequency,
        quietHoursEnabled: form.quietHoursEnabled,
        quietStart: form.quietStart,
        quietEnd: form.quietEnd,
        createTime: s.createTime,
        updateTime: s.updateTime,
      );
    });
    try {
      final service = ref.read(notificationSettingsServiceProvider);
      final updated = await service.updateSettings(form);
      if (!mounted) return;
      setState(() => _settings = updated);
    } catch (e) {
      // 回滚
      if (!mounted) return;
      setState(() => _settings = s);
      showError(context, '保存失败: $e');
    }
  }

  Future<void> _pickTime({required bool isStart}) async {
    final s = _settings;
    if (s == null) return;
    final initialStr = isStart ? s.quietStart : s.quietEnd;
    TimeOfDay initial;
    if (initialStr != null && initialStr.isNotEmpty) {
      final parts = initialStr.split(':');
      initial = TimeOfDay(
        hour: int.tryParse(parts[0]) ?? 22,
        minute: int.tryParse(parts.length > 1 ? parts[1] : '0') ?? 0,
      );
    } else {
      initial = isStart ? const TimeOfDay(hour: 22, minute: 0) : const TimeOfDay(hour: 8, minute: 0);
    }
    final picked = await showTimePicker(context: context, initialTime: initial);
    if (picked == null) return;
    final timeStr =
        '${picked.hour.toString().padLeft(2, '0')}:${picked.minute.toString().padLeft(2, '0')}';
    await _updateSetting((s) => NotificationSettingsForm(
          systemEnabled: s.systemEnabled,
          predictionEnabled: s.predictionEnabled,
          activityEnabled: s.activityEnabled,
          announcementEnabled: s.announcementEnabled,
          emailEnabled: s.emailEnabled,
          smsEnabled: s.smsEnabled,
          pushEnabled: s.pushEnabled,
          digestEnabled: s.digestEnabled,
          digestFrequency: s.digestFrequency,
          quietHoursEnabled: s.quietHoursEnabled,
          quietStart: isStart ? timeStr : s.quietStart,
          quietEnd: isStart ? s.quietEnd : timeStr,
        ));
  }

  String _formatTime(String? timeStr) {
    if (timeStr == null || timeStr.isEmpty) return '--:--';
    return timeStr;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('系统设置')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // 账号安全
          _buildSection(theme, '账号', [
            _buildTile(Icons.person_outline, '个人信息', () {}),
            _buildTile(Icons.lock_outline, '修改密码', () {}),
          ]),
          const SizedBox(height: 16),
          // 通知设置（从 API 加载）
          if (_isLoading)
            _buildSection(theme, '通知', [
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 24),
                child: Center(child: CircularProgressIndicator()),
              ),
            ])
          else if (_error != null)
            _buildSection(theme, '通知', [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    Text(_error!, style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.error)),
                    const SizedBox(height: 8),
                    TextButton.icon(
                      onPressed: _loadSettings,
                      icon: const Icon(Icons.refresh, size: 18),
                      label: const Text('重试'),
                    ),
                  ],
                ),
              ),
            ])
          else ...[
            _buildSection(theme, '通知类型', [
              _buildSwitchTile('系统通知', '系统公告与维护通知', _settings!.systemEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, systemEnabled: v))),
              _buildSwitchTile('处理通知', '图像处理完成时通知', _settings!.predictionEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, predictionEnabled: v))),
              _buildSwitchTile('活动通知', '优惠活动与促销信息', _settings!.activityEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, activityEnabled: v))),
              _buildSwitchTile('公告通知', '系统公告通知', _settings!.announcementEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, announcementEnabled: v))),
              _buildSwitchTile('邮件通知', '接收邮件通知', _settings!.emailEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, emailEnabled: v))),
              _buildSwitchTile('短信通知', '接收短信通知', _settings!.smsEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, smsEnabled: v))),
              _buildSwitchTile('推送通知', '接收应用推送通知', _settings!.pushEnabled,
                  (v) => _updateSetting((s) => _formFrom(s, pushEnabled: v))),
            ]),
            const SizedBox(height: 20),
            // 免打扰
            _buildSection(theme, '免打扰', [
              SwitchListTile(
                secondary: const Icon(Icons.nightlight_round),
                title: const Text('开启免打扰'),
                subtitle: const Text('在指定时段内不接收通知'),
                value: _settings!.quietHoursEnabled,
                onChanged: (v) => _updateSetting((s) => _formFrom(s, quietHoursEnabled: v)),
              ),
              if (_settings!.quietHoursEnabled) ...[
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  leading: const Icon(Icons.bedtime_outlined),
                  title: const Text('开始时间'),
                  trailing: Text(
                    _formatTime(_settings!.quietStart),
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  onTap: () => _pickTime(isStart: true),
                ),
                const Divider(height: 1, indent: 16, endIndent: 16),
                ListTile(
                  leading: const Icon(Icons.wb_sunny_outlined),
                  title: const Text('结束时间'),
                  trailing: Text(
                    _formatTime(_settings!.quietEnd),
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.primary,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  onTap: () => _pickTime(isStart: false),
                ),
              ],
            ]),
          ],
          const SizedBox(height: 16),
          // 缓存
          _buildSection(theme, '存储', [
            _buildTile(Icons.cleaning_services_outlined, '清除缓存', () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('缓存已清除')),
              );
            }),
          ]),
          const SizedBox(height: 16),
          // 关于
          _buildSection(theme, '其他', [
            _buildTile(Icons.info_outline, '关于我们', () => context.push(AppRouterConfig.about)),
            _buildTile(Icons.description_outlined, '用户协议', () {}),
            _buildTile(Icons.privacy_tip_outlined, '隐私政策', () {}),
          ]),
          const SizedBox(height: 32),
          SizedBox(
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
          ),
        ],
      ),
    );
  }

  /// 从当前 NotificationSettings 构建 NotificationSettingsForm，可选覆盖字段
  NotificationSettingsForm _formFrom(
    NotificationSettings s, {
    bool? systemEnabled,
    bool? predictionEnabled,
    bool? activityEnabled,
    bool? announcementEnabled,
    bool? emailEnabled,
    bool? smsEnabled,
    bool? pushEnabled,
    bool? digestEnabled,
    String? digestFrequency,
    bool? quietHoursEnabled,
    String? quietStart,
    String? quietEnd,
  }) {
    return NotificationSettingsForm(
      systemEnabled: systemEnabled ?? s.systemEnabled,
      predictionEnabled: predictionEnabled ?? s.predictionEnabled,
      activityEnabled: activityEnabled ?? s.activityEnabled,
      announcementEnabled: announcementEnabled ?? s.announcementEnabled,
      emailEnabled: emailEnabled ?? s.emailEnabled,
      smsEnabled: smsEnabled ?? s.smsEnabled,
      pushEnabled: pushEnabled ?? s.pushEnabled,
      digestEnabled: digestEnabled ?? s.digestEnabled,
      digestFrequency: digestFrequency ?? s.digestFrequency,
      quietHoursEnabled: quietHoursEnabled ?? s.quietHoursEnabled,
      quietStart: quietStart ?? s.quietStart,
      quietEnd: quietEnd ?? s.quietEnd,
    );
  }

  Widget _buildSection(ThemeData theme, String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(
            title,
            style: theme.textTheme.labelMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
        ),
        Container(
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: theme.dividerColor),
          ),
          child: Column(children: children),
        ),
      ],
    );
  }

  Widget _buildTile(IconData icon, String title, VoidCallback onTap) {
    return ListTile(
      leading: Icon(icon, size: 22),
      title: Text(title),
      trailing: const Icon(Icons.chevron_right, size: 20),
      onTap: onTap,
      visualDensity: VisualDensity.compact,
    );
  }

  Widget _buildSwitchTile(
    String title,
    String subtitle,
    bool value,
    ValueChanged<bool> onChanged,
  ) {
    return SwitchListTile(
      secondary: const Icon(Icons.notifications_outlined),
      title: Text(title),
      subtitle: Text(subtitle),
      value: value,
      onChanged: onChanged,
    );
  }
}
