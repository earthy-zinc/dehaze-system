import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../theme/app_theme.dart';
import '../../utils/ui_utils.dart';

/// 消息设置 — L2 页面
///
/// 通知开关 + 免打扰时段，对接 NotificationSettingService
class NotifyPage extends ConsumerStatefulWidget {
  const NotifyPage({super.key});

  @override
  ConsumerState<NotifyPage> createState() => _NotifyPageState();
}

class _NotifyPageState extends ConsumerState<NotifyPage> {
  bool _pushEnabled = true;
  bool _systemEnabled = true;
  bool _processDone = true;
  bool _activityEnabled = true;
  bool _dndEnabled = false;
  TimeOfDay _dndStart = const TimeOfDay(hour: 22, minute: 0);
  TimeOfDay _dndEnd = const TimeOfDay(hour: 8, minute: 0);
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() => _isLoading = true);
    // TODO: 对接 NotificationSettingService.get 真实 API
    await Future<void>.delayed(const Duration(milliseconds: 300));
    if (!mounted) return;
    setState(() => _isLoading = false);
  }

  Future<void> _save() async {
    // TODO: 对接 NotificationSettingService.update 真实 API
    showSnackBar(context, '设置已保存');
  }

  Future<void> _pickTime(bool isStart) async {
    final picked = await showTimePicker(
      context: context,
      initialTime: isStart ? _dndStart : _dndEnd,
    );
    if (picked != null) {
      setState(() {
        if (isStart) {
          _dndStart = picked;
        } else {
          _dndEnd = picked;
        }
      });
      _save();
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text('消息设置')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('消息设置')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // 通知开关
          _buildSection(theme, '通知类型', [
            _buildSwitchTile('推送通知', '接收应用推送通知', _pushEnabled, (v) => setState(() { _pushEnabled = v; _save(); })),
            _buildSwitchTile('系统通知', '系统公告与维护通知', _systemEnabled, (v) => setState(() { _systemEnabled = v; _save(); })),
            _buildSwitchTile('处理完成', '图像处理完成时通知', _processDone, (v) => setState(() { _processDone = v; _save(); })),
            _buildSwitchTile('活动通知', '优惠活动与促销信息', _activityEnabled, (v) => setState(() { _activityEnabled = v; _save(); })),
          ]),
          const SizedBox(height: 20),
          // 免打扰
          _buildSection(theme, '免打扰', [
            SwitchListTile(
              secondary: const Icon(Icons.nightlight_round),
              title: const Text('开启免打扰'),
              subtitle: const Text('在指定时段内不接收通知'),
              value: _dndEnabled,
              onChanged: (v) {
                setState(() => _dndEnabled = v);
                _save();
              },
            ),
            if (_dndEnabled) ...[
              const Divider(height: 1, indent: 16, endIndent: 16),
              ListTile(
                leading: const Icon(Icons.bedtime_outlined),
                title: const Text('开始时间'),
                trailing: Text(_dndStart.format(context), style: TextStyle(color: AppTheme.brandBlue, fontWeight: FontWeight.w600)),
                onTap: () => _pickTime(true),
              ),
              const Divider(height: 1, indent: 16, endIndent: 16),
              ListTile(
                leading: const Icon(Icons.wb_sunny_outlined),
                title: const Text('结束时间'),
                trailing: Text(_dndEnd.format(context), style: TextStyle(color: AppTheme.brandBlue, fontWeight: FontWeight.w600)),
                onTap: () => _pickTime(false),
              ),
            ],
          ]),
        ],
      ),
    );
  }

  Widget _buildSection(ThemeData theme, String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(title, style: theme.textTheme.labelMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          )),
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

  Widget _buildSwitchTile(String title, String subtitle, bool value, ValueChanged<bool> onChanged) {
    return SwitchListTile(
      secondary: const Icon(Icons.notifications_outlined),
      title: Text(title),
      subtitle: Text(subtitle),
      value: value,
      onChanged: onChanged,
    );
  }
}
