import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../layout/main_layout.dart';
import '../pages/dataset/index.dart';
import '../pages/home/index.dart';
import '../pages/image_input/index.dart';

/// 应用路由配置
///
/// 统一管理所有路由路径和导航逻辑
/// 与 menu_config.dart 菜单配置保持一致
class AppRouterConfig {
  // ==================== 基础路由 ====================
  static const String home = '/home';
  static const String splash = '/splash';
  static const String dehaze = '/dehaze';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';
  static const String settings = '/settings';
  static const String about = '/about';

  // ==================== 处理流程路由 ====================
  static const String imageInput = '/image-input';
  static const String algorithmSelect = '/algorithm-select';
  static const String processing = '/processing';

  // ==================== 效果对比路由 ====================
  static const String sideBySide = '/side-by-side';
  static const String overlay = '/overlay';
  static const String magnifier = '/magnifier';
  static const String filter = '/filter';
  static const String metrics = '/metrics';
  static const String algorithm = '/algorithm';

  // ==================== 数据管理路由 ====================
  static const String dataset = '/dataset';
  static const String datasetDetail = '/dataset/:id';

  /// 获取数据集详情路由
  static String getDatasetDetailPath(int id) => '/dataset/$id';

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
          // 数据集管理
          GoRoute(
            path: dataset,
            name: 'dataset',
            builder: (context, state) => const DatasetPage(),
            routes: [
              // 数据集详情（嵌套路由）
              GoRoute(
                path: ':id',
                name: 'dataset-detail',
                builder: (context, state) {
                  final id = state.pathParameters['id'];
                  return DatasetPage(initialDatasetId: int.tryParse(id ?? ''));
                },
              ),
            ],
          ),
          // 图像输入
          GoRoute(
            path: imageInput,
            name: 'image-input',
            builder: (context, state) => const ImageInputPage(),
          ),
          // TODO: 添加其他功能页面路由
        ],
      ),
    ],
    errorBuilder: (context, state) => Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: Theme.of(context).colorScheme.error,
            ),
            const SizedBox(height: 16),
            Text(
              '页面未找到',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              '路径: ${state.uri}',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () => context.go(home),
              icon: const Icon(Icons.home),
              label: const Text('返回首页'),
            ),
          ],
        ),
      ),
    ),
    redirect: (context, state) {
      // 全局重定向逻辑
      if (state.fullPath == splash || state.fullPath == '/') {
        return home;
      }
      return null;
    },
  );

  static GoRouter get router => _router;

  /// 检查路由是否为当前活跃路由
  static bool isActiveRoute(BuildContext context, String route) {
    final currentLocation = GoRouterState.of(context).uri.toString();
    return currentLocation.startsWith(route);
  }
}
