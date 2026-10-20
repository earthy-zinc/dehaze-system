import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

/// L3 深度沉浸页骨架
///
/// 全屏 Scaffold，无 AppBar/无底部导航，顶部深色半透明工具栏（返回 + 标题 + 右侧操作区）。
/// 适用于效果对比 5 模式（并排/重叠/放大镜/滤镜/指标）等全屏审阅场景。
class ImmersiveScaffold extends StatelessWidget {
  const ImmersiveScaffold({
    super.key,
    required this.title,
    this.actions = const [],
    required this.body,
  });

  final String title;
  final List<Widget> actions;
  final Widget body;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: body),
          // 顶部深色半透明工具栏
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 56 + MediaQuery.of(context).padding.top,
              padding: EdgeInsets.only(top: MediaQuery.of(context).padding.top),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.black.withValues(alpha: 0.65),
                    Colors.black.withValues(alpha: 0.0),
                  ],
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                    onPressed: () {
                      if (context.canPop()) {
                        context.pop();
                      }
                    },
                  ),
                  Expanded(
                    child: Text(
                      title,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  ...actions,
                  const SizedBox(width: 4),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
