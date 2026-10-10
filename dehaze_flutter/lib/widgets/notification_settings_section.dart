import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/notification_settings_model.dart';
import '../providers/providers.dart';
import '../theme/app_theme.dart';
import '../utils/ui_utils.dart';

/// 通知偏好设置区块
///
/// 系统设置（页头齿轮）与消息设置（消息 → 设置）展示的是同一份偏好，
/// 因此共用一个组件，避免两处各自维护字段映射。
///
/// 后端 PATCH 语义为 preferences 逐 key 深合并，每次变更提交完整偏好。
class NotificationSettingsSection extends ConsumerStatefulWidget {
  const NotificationSettingsSection({super.key});

  @override
  ConsumerState<NotificationSettingsSection> createState() =>
      _NotificationSettingsSectionState();
}

class _NotificationSettingsSectionState
    extends ConsumerState<NotificationSettingsSection> {
  /// 消息类型推送开关（preferences.typeChannels）
  static const _typeChannels = [
    (key: 'announcement', title: '系统公告', subtitle: '系统维护、功能更新与活动公告'),
    (key: 'business', title: '业务通知', subtitle: '订单状态、任务完成与反馈回复'),
    (key: 'member', title: '会员通知', subtitle: '等级变更、到期预警与权益更新'),
  ];

  /// 业务模块开关（preferences.moduleSwitches）
  static const _moduleSwitches = [
    (key: 'prediction', title: '去雾处理', subtitle: '去雾任务完成或失败时通知'),
    (key: 'feedback', title: '反馈评价', subtitle: '反馈回复与评价处理进展'),
    (key: 'announcement', title: '系统公告', subtitle: '管理员发送的公告消息'),
  ];

  static const _defaultDndStart = '22:00:00';
  static const _defaultDndEnd = '08:00:00';

  NotificationSettings? _settings;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final settings = await ref
          .read(notificationSettingsServiceProvider)
          .getSettings();
      if (!mounted) return;
      setState(() {
        _settings = settings;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  /// 乐观更新后提交，失败回滚到变更前状态
  Future<void> _save(NotificationSettings next) async {
    final previous = _settings;
    if (previous == null) return;
    setState(() => _settings = next);
    try {
      final updated = await ref
          .read(notificationSettingsServiceProvider)
          .updateSettings(next);
      if (!mounted) return;
      setState(() => _settings = updated);
    } catch (e) {
      if (!mounted) return;
      setState(() => _settings = previous);
      showError(context, '保存失败: $e');
    }
  }

  Future<void> _pickTime({required bool isStart}) async {
    final settings = _settings;
    if (settings == null) return;
    final current = isStart ? settings.dndStart : settings.dndEnd;
    final picked = await showTimePicker(
      context: context,
      initialTime: _parseTime(current, isStart ? 22 : 8),
    );
    if (picked == null) return;
    final value = '${_two(picked.hour)}:${_two(picked.minute)}:00';
    await _save(
      settings.copyWith(
        dndStart: isStart ? value : settings.dndStart,
        dndEnd: isStart ? settings.dndEnd : value,
      ),
    );
  }

  static TimeOfDay _parseTime(String? time, int fallbackHour) {
    final parts = time?.split(':');
    if (parts == null || parts.isEmpty) {
      return TimeOfDay(hour: fallbackHour, minute: 0);
    }
    return TimeOfDay(
      hour: int.tryParse(parts[0]) ?? fallbackHour,
      minute: parts.length > 1 ? int.tryParse(parts[1]) ?? 0 : 0,
    );
  }

  static String _two(int value) => value.toString().padLeft(2, '0');

  static String _formatTime(String? time) =>
      time == null || time.isEmpty ? '--:--' : time.substring(0, 5);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (_isLoading) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 24),
        child: Center(child: CircularProgressIndicator()),
      );
    }
    if (_error != null) {
      return Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(
              _error!,
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.error,
              ),
            ),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed: _load,
              icon: const Icon(Icons.refresh, size: 18),
              label: const Text('重试'),
            ),
          ],
        ),
      );
    }

    final settings = _settings!;
    final preferences = settings.preferences;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildGroup(theme, '推送', [
          SwitchListTile(
            secondary: const Icon(Icons.notifications_outlined),
            title: const Text('推送通知'),
            subtitle: const Text('接收应用推送通知'),
            value: settings.pushEnabled,
            onChanged: (v) => _save(settings.copyWith(pushEnabled: v)),
          ),
        ]),
        const SizedBox(height: 16),
        _buildGroup(theme, '按类型设置推送', [
          for (final channel in _typeChannels)
            SwitchListTile(
              secondary: const Icon(Icons.label_outline),
              title: Text(channel.title),
              subtitle: Text(channel.subtitle),
              value: preferences.isTypeChannelPushEnabled(channel.key),
              onChanged: settings.pushEnabled
                  ? (v) => _save(
                      settings.copyWith(
                        preferences: preferences.withTypeChannel(
                          channel.key,
                          v,
                        ),
                      ),
                    )
                  : null,
            ),
        ]),
        const SizedBox(height: 16),
        _buildGroup(theme, '按消息模块', [
          for (final module in _moduleSwitches)
            SwitchListTile(
              secondary: const Icon(Icons.widgets_outlined),
              title: Text(module.title),
              subtitle: Text(module.subtitle),
              value: preferences.isModuleEnabled(module.key),
              onChanged: (v) => _save(
                settings.copyWith(
                  preferences: preferences.withModuleSwitch(module.key, v),
                ),
              ),
            ),
        ]),
        const SizedBox(height: 16),
        _buildGroup(theme, '免打扰', [
          SwitchListTile(
            secondary: const Icon(Icons.nightlight_round),
            title: const Text('开启免打扰'),
            subtitle: const Text('在指定时段内不接收通知'),
            value: settings.dndEnabled,
            onChanged: (v) => _save(
              settings.copyWith(
                dndEnabled: v,
                dndStart: v ? settings.dndStart ?? _defaultDndStart : null,
                dndEnd: v ? settings.dndEnd ?? _defaultDndEnd : null,
              ),
            ),
          ),
          if (settings.dndEnabled) ...[
            const Divider(height: 1, indent: 16, endIndent: 16),
            ListTile(
              leading: const Icon(Icons.bedtime_outlined),
              title: const Text('开始时间'),
              trailing: Text(
                _formatTime(settings.dndStart),
                style: const TextStyle(
                  color: AppTheme.brandBlue,
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
                _formatTime(settings.dndEnd),
                style: const TextStyle(
                  color: AppTheme.brandBlue,
                  fontWeight: FontWeight.w600,
                ),
              ),
              onTap: () => _pickTime(isStart: false),
            ),
          ],
        ]),
      ],
    );
  }

  Widget _buildGroup(ThemeData theme, String title, List<Widget> children) {
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
            borderRadius: BorderRadius.circular(AppTheme.radiusM),
            border: Border.all(color: theme.dividerColor),
          ),
          child: Column(children: children),
        ),
      ],
    );
  }
}
