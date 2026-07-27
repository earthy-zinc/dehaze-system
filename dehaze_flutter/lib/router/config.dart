import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../providers/auth_provider.dart';
import '../layout/main_layout.dart';
import '../pages/algorithm_select/index.dart';
import '../pages/comparison/algorithm_info.dart';
import '../pages/comparison/filter.dart';
import '../pages/comparison/magnifier.dart';
import '../pages/comparison/metrics.dart';
import '../pages/comparison/overlay.dart';
import '../pages/comparison/side_by_side.dart';
import '../pages/dataset/index.dart';
import '../pages/home/index.dart';
import '../pages/image_input/index.dart';
import '../pages/login/index.dart';
import '../pages/processing/index.dart';
import '../pages/profile/index.dart';
import '../pages/register/index.dart';
import '../pages/task_history/index.dart';

/// 应用路由配置
///
/// 统一管理所有路由路径和导航逻辑
/// 与 menu_config.dart 菜单配置保持一致
class AppRouterConfig {
  const AppRouterConfig._();

  // ==================== 路由路径常量 ====================

  // 基础路由
  static const String home = '/home';
  static const String login = '/login';
  static const String register = '/register';
  static const String profile = '/profile';

  // 处理流程路由
  static const String imageInput = '/image-input';
  static const String algorithmSelect = '/algorithm-select';
  static const String processing = '/processing';

  // 效果对比路由
  static const String sideBySide = '/side-by-side';
  static const String overlay = '/overlay';
  static const String magnifier = '/magnifier';
  static const String filter = '/filter';
  static const String metrics = '/metrics';
  static const String algorithm = '/algorithm';

  // 数据管理路由
  static const String dataset = '/dataset';
  static const String datasetDetail = '/dataset/:id';

  // 历史记录路由
  static const String taskHistory = '/task-history';

  /// 获取数据集详情路由
  static String getDatasetDetailPath(int id) => '/dataset/$id';

  /// 登录态白名单（无需登录即可访问）
  static const List<String> publicRoutes = [login, register, home];

  /// 检查路由是否为当前活跃路由
  static bool isActiveRoute(BuildContext context, String route) {
    final currentLocation = GoRouterState.of(context).uri.toString();
    return currentLocation.startsWith(route);
  }
}

/// GoRouter Provider
///
/// 监听认证状态，自动重定向：
/// - 未登录访问受保护路由 → 跳转登录页
/// - 已登录访问登录页 → 跳转首页
final goRouterProvider = Provider<GoRouter>((ref) {
  final authState = ref.watch(authProvider);

  return GoRouter(
    initialLocation: AppRouterConfig.home,
    debugLogDiagnostics: true,
    routes: [
      // 登录页（无 ShellRoute 包裹）
      GoRoute(
        path: AppRouterConfig.login,
        name: 'login',
        builder: (context, state) => const LoginPage(),
      ),
      // 注册页（无 ShellRoute 包裹）
      GoRoute(
        path: AppRouterConfig.register,
        name: 'register',
        builder: (context, state) => const RegisterPage(),
      ),
      // 主布局路由
      ShellRoute(
        builder: (context, state, child) => MainLayout(child: child),
        routes: [
          // 首页
          GoRoute(
            path: AppRouterConfig.home,
            name: 'home',
            builder: (context, state) => const HomePage(),
          ),
          // 数据集管理
          GoRoute(
            path: AppRouterConfig.dataset,
            name: 'dataset',
            builder: (context, state) => const DatasetPage(),
            routes: [
              GoRoute(
                path: ':id',
                name: 'dataset-detail',
                builder: (context, state) {
                  final id = state.pathParameters['id'];
                  return DatasetPage(
                      initialDatasetId: int.tryParse(id ?? ''));
                },
              ),
            ],
          ),
          // 图像输入
          GoRoute(
            path: AppRouterConfig.imageInput,
            name: 'image-input',
            builder: (context, state) => const ImageInputPage(),
          ),
          // 算法选择
          GoRoute(
            path: AppRouterConfig.algorithmSelect,
            name: 'algorithm-select',
            builder: (context, state) => const AlgorithmSelectPage(),
          ),
          // 去雾处理
          GoRoute(
            path: AppRouterConfig.processing,
            name: 'processing',
            builder: (context, state) => const ProcessingPage(),
          ),
          // 效果对比页面
          GoRoute(
            path: AppRouterConfig.sideBySide,
            name: 'side-by-side',
            builder: (context, state) => const SideBySidePage(),
          ),
          GoRoute(
            path: AppRouterConfig.overlay,
            name: 'overlay',
            builder: (context, state) => const OverlayPage(),
          ),
          GoRoute(
            path: AppRouterConfig.magnifier,
            name: 'magnifier',
            builder: (context, state) => const MagnifierPage(),
          ),
          GoRoute(
            path: AppRouterConfig.filter,
            name: 'filter',
            builder: (context, state) => const FilterPage(),
          ),
          GoRoute(
            path: AppRouterConfig.metrics,
            name: 'metrics',
            builder: (context, state) => const MetricsPage(),
          ),
          GoRoute(
            path: AppRouterConfig.algorithm,
            name: 'algorithm-info',
            builder: (context, state) => const AlgorithmInfoPage(),
          ),
          // 用户中心
          GoRoute(
            path: AppRouterConfig.profile,
            name: 'profile',
            builder: (context, state) => const ProfilePage(),
          ),
          // 处理历史
          GoRoute(
            path: AppRouterConfig.taskHistory,
            name: 'task-history',
            builder: (context, state) => const TaskHistoryPage(),
          ),
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
              onPressed: () => context.go(AppRouterConfig.home),
              icon: const Icon(Icons.home),
              label: const Text('返回首页'),
            ),
          ],
        ),
      ),
    ),
    redirect: (context, state) {
      final isLoggedIn = authState.isAuthenticated;
      final isGoingToAuthPage = state.matchedLocation == AppRouterConfig.login ||
          state.matchedLocation == AppRouterConfig.register;
      final isPublicRoute =
          AppRouterConfig.publicRoutes.contains(state.matchedLocation);

      // 未登录访问受保护路由 → 跳转登录
      if (!isLoggedIn && !isPublicRoute && !isGoingToAuthPage) {
        return AppRouterConfig.login;
      }

      // 已登录访问登录/注册页 → 跳转首页
      if (isLoggedIn && isGoingToAuthPage) {
        return AppRouterConfig.home;
      }

      return null;
    },
  );
});
