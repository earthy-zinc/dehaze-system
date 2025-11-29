import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../layout/main_layout.dart';
import '../pages/home/index.dart';

class AppRouterConfig {
  static const String home = '/home';
  static const String splash = '/splash';
  static const String dehaze = '/dehaze';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String about = '/about';

  // 侧边菜单路由
  static const String imageInput = '/image-input';
  static const String algorithmSelect = '/algorithm-select';
  static const String processing = '/processing';
  static const String sideBySide = '/side-by-side';
  static const String overlay = '/overlay';
  static const String magnifier = '/magnifier';
  static const String filter = '/filter';
  static const String metrics = '/metrics';
  static const String algorithm = '/algorithm';
  static const String dataset = '/dataset';

  static final GoRouter _router = GoRouter(
    initialLocation: home,
    debugLogDiagnostics: true,
    routes: [
      ShellRoute(
        builder: (context, state, child) => MainLayout(child: child),
        routes: [
          // 首页 - 主页面
          GoRoute(
            path: home,
            name: 'home',
            builder: (context, state) => const HomePage(),
          ),
        ],
      ),
    ],
    errorBuilder: (context, state) => Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('页面未找到'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => context.go('/home'),
              child: const Text('返回首页'),
            ),
          ],
        ),
      ),
    ),
    redirect: (context, state) {
      // 在这里可以添加全局重定向逻辑，例如认证检查等
      if (state.fullPath == splash) {
        return home;
      }
      return null; // 不进行重定向
    },
  );

  static GoRouter get router => _router;
}
