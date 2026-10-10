import 'package:flutter/material.dart';

import '../../widgets/notification_settings_section.dart';

/// 消息设置 — L2 页面
///
/// 通知开关 + 免打扰时段，对接 NotificationSettingsService（GET / PATCH）。
class NotifyPage extends StatelessWidget {
  const NotifyPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('消息设置')),
      body: const SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: NotificationSettingsSection(),
      ),
    );
  }
}
