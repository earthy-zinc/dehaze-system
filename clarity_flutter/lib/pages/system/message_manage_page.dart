import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'announcement_tab.dart';
import 'message_list_tab.dart';
import 'message_template_tab.dart';

/// 消息管理页面（L2）
///
/// 权限：sys:notify:*
class MessageManagePage extends ConsumerStatefulWidget {
  const MessageManagePage({super.key});

  @override
  ConsumerState<MessageManagePage> createState() => _MessageManagePageState();
}

class _MessageManagePageState extends ConsumerState<MessageManagePage>
    with SingleTickerProviderStateMixin {
  late TabController _tabCtrl;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('消息管理'),
        bottom: TabBar(
          controller: _tabCtrl,
          tabs: const [
            Tab(text: '消息列表'),
            Tab(text: '公告管理'),
            Tab(text: '消息模板'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: const [
          MessageListTab(),
          AnnouncementTab(),
          MessageTemplateTab(),
        ],
      ),
    );
  }
}
