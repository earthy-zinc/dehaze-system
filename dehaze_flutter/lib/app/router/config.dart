import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../features/dehaze/presentation/pages/dehaze_page.dart';
import '../../features/home/presentation/pages/home_page.dart';

class AppRouterConfig {
  static const String splash = '/splash';
  static const String home = '/home';
  static const String dehaze = '/dehaze';
  static const String auth = '/auth';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String about = '/about';

  static final GoRouter _router = GoRouter(
    initialLocation: '/home',
    debugLogDiagnostics: true,
    routes: [
      // 首页 - 主页面
      GoRoute(
        path: home,
        name: 'home_main',
        builder: (context, state) => const HomePage(),
      ),
      // 去雾页面
      GoRoute(
        path: dehaze,
        name: 'dehaze',
        builder: (context, state) => const DehazePage(),
      ),

      // 认证相关路由
      GoRoute(
        path: auth,
        name: 'auth',
        builder: (context, state) => const DehazePage(),
        routes: [
          GoRoute(
            path: login,
            name: 'login',
            builder: (context, state) => const DehazePage(),
          ),
          GoRoute(
            path: register,
            name: 'register',
            builder: (context, state) => const DehazePage(),
          ),
        ],
      ),

      // 其他路由占位符 - 目前都指向DehazePage
      GoRoute(
        path: splash,
        name: 'splash_page',
        builder: (context, state) => const DehazePage(),
      ),
      GoRoute(
        path: profile,
        name: 'profile',
        builder: (context, state) => const DehazePage(),
      ),
      GoRoute(
        path: settings,
        name: 'settings',
        builder: (context, state) => const DehazePage(),
      ),
      GoRoute(
        path: about,
        name: 'about',
        builder: (context, state) => const DehazePage(),
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
  );

  static GoRouter get router => _router;
}
