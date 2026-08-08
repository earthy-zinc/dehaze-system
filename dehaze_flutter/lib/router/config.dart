import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../providers/auth_provider.dart';
import '../layout/main_layout.dart';
import '../pages/algorithm/index.dart';
import '../pages/algorithm_select/index.dart';
import '../pages/batch/index.dart';
import '../pages/comparison/algorithm_info.dart';
import '../pages/comparison/filter.dart';
import '../pages/comparison/magnifier.dart';
import '../pages/comparison/metrics.dart';
import '../pages/comparison/overlay.dart';
import '../pages/comparison/side_by_side.dart';
import '../pages/dataset/index.dart';
import '../pages/dehaze/index.dart';
import '../pages/home/index.dart';
import '../pages/image_input/index.dart';
import '../pages/login/index.dart';
import '../pages/metrics_manage/index.dart';
import '../pages/messages/index.dart';
import '../pages/messages/detail/index.dart';
import '../pages/notify/notify_page.dart';
import '../pages/personal/about_page.dart';
import '../pages/personal/favorites_page.dart';
import '../pages/personal/feedback_page.dart';
import '../pages/personal/files_page.dart';
import '../pages/personal/help_page.dart';
import '../pages/personal/member_page.dart';
import '../pages/personal/orders_page.dart';
import '../pages/personal/package_page.dart';
import '../pages/personal/quota_page.dart';
import '../pages/personal/settings_page.dart';
import '../pages/processing/index.dart';
import '../pages/profile/index.dart';
import '../pages/register/index.dart';
import '../pages/task_history/index.dart';
import '../pages/tools/index.dart';
import '../pages/dashboard/index.dart';
import '../pages/system/user_page.dart';
import '../pages/system/role_page.dart';
import '../pages/system/menu_page.dart';
import '../pages/system/dept_page.dart';
import '../pages/system/dict_page.dart';
import '../pages/system/algorithm_manage_page.dart';
import '../pages/system/dataset_manage_page.dart';
import '../pages/system/task_manage_page.dart';
import '../pages/system/member_manage_page.dart';
import '../pages/system/package_manage_page.dart';
import '../pages/system/order_manage_page.dart';
import '../pages/system/feedback_manage_page.dart';
import '../pages/system/message_manage_page.dart';
import '../pages/system/recommend_manage_page.dart';

/// 应用路由配置
class AppRouterConfig {
  const AppRouterConfig._();

  // ==================== 路由路径常量 ====================

  // 认证
  static const String login = '/login';
  static const String register = '/register';

  // Tab 根路径
  static const String home = '/home';
  static const String tools = '/tools';
  static const String dehaze = '/dehaze';
  static const String messages = '/messages';
  static const String profile = '/profile';

  // L2 子页面（完整路径，用于 context.go()）
  static const String imageInput = '/tools/image-input';
  static const String algorithmBrowse = '/tools/algorithm-browse';
  static const String algorithmSelect = '/dehaze/algorithm-select';
  static const String dataset = '/tools/dataset';
  static const String processing = '/dehaze/processing';
  static const String taskHistory = '/profile/task-history';
  static const String batch = '/tools/batch';
  static const String metricsManage = '/tools/metrics-manage';

  // Profile L2 子页面
  static const String files = '/profile/files';
  static const String orders = '/profile/orders';
  static const String quota = '/profile/quota';
  static const String member = '/profile/member';
  static const String package = '/profile/package';
  static const String feedback = '/profile/feedback';
  static const String favorites = '/profile/favorites';
  static const String settings = '/profile/settings';
  static const String help = '/profile/help';
  static const String about = '/profile/about';
  static const String notify = '/profile/notify';

  // L3 沉浸页（ShellRoute 外）
  static const String sideBySide = '/compare/side-by-side';
  static const String overlay = '/compare/overlay';
  static const String magnifier = '/compare/magnifier';
  static const String filter = '/compare/filter';
  static const String compareMetrics = '/compare/metrics';
  static const String algorithm = '/algorithm';

  /// 登录态白名单
  static const List<String> publicRoutes = [login, register, home];
}

/// GoRouter Provider
final goRouterProvider = Provider<GoRouter>((ref) {
  final authState = ref.watch(authProvider);

  return GoRouter(
    initialLocation: AppRouterConfig.home,
    debugLogDiagnostics: true,
    routes: [
      // ==================== L0 认证页（ShellRoute 外） ====================
      GoRoute(
        path: AppRouterConfig.login,
        name: 'login',
        builder: (context, state) => const LoginPage(),
      ),
      GoRoute(
        path: AppRouterConfig.register,
        name: 'register',
        builder: (context, state) => const RegisterPage(),
      ),

      // ==================== L3 沉浸页（ShellRoute 外） ====================
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
        path: AppRouterConfig.compareMetrics,
        name: 'compare-metrics',
        builder: (context, state) => const MetricsPage(),
      ),
      GoRoute(
        path: '/algorithm',
        name: 'algorithm-info',
        builder: (context, state) => const AlgorithmInfoPage(),
      ),

      // ==================== StatefulShellRoute (5 Tab) ====================
      StatefulShellRoute.indexedStack(
        builder: (context, state, navigationShell) =>
            MainLayout(navigationShell: navigationShell),
        branches: [
          // Branch 0: 首页
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: AppRouterConfig.home,
                name: 'home',
                builder: (context, state) => const HomePage(),
              ),
            ],
          ),
          // Branch 1: 工具
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: AppRouterConfig.tools,
                name: 'tools',
                builder: (context, state) => const ToolsPage(),
                routes: [
                  GoRoute(
                    path: 'image-input',
                    name: 'image-input',
                    builder: (context, state) => const ImageInputPage(),
                  ),
                  GoRoute(
                    path: 'algorithm-browse',
                    name: 'algorithm-browse',
                    builder: (context, state) => const AlgorithmBrowsePage(),
                  ),
                  GoRoute(
                    path: 'dataset',
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
                  GoRoute(
                    path: 'batch',
                    name: 'batch',
                    builder: (context, state) => const BatchPage(),
                  ),
                  GoRoute(
                    path: 'metrics-manage',
                    name: 'metrics-manage',
                    builder: (context, state) => const MetricsManagePage(),
                  ),
                ],
              ),
            ],
          ),
          // Branch 2: 去雾
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: AppRouterConfig.dehaze,
                name: 'dehaze',
                builder: (context, state) => const DehazePage(),
                routes: [
                  GoRoute(
                    path: 'algorithm-select',
                    name: 'dehaze-algorithm-select',
                    builder: (context, state) => const AlgorithmSelectPage(),
                  ),
                  GoRoute(
                    path: 'processing',
                    name: 'dehaze-processing',
                    builder: (context, state) => const ProcessingPage(),
                  ),
                ],
              ),
            ],
          ),
          // Branch 3: 消息
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: AppRouterConfig.messages,
                name: 'messages',
                builder: (context, state) => const MessagesPage(),
                routes: [
                  GoRoute(
                    path: ':id',
                    name: 'message-detail',
                    builder: (context, state) {
                      final id = int.tryParse(state.pathParameters['id'] ?? '') ?? 0;
                      return MessageDetailPage(messageId: id);
                    },
                  ),
                  GoRoute(
                    path: 'notify',
                    name: 'message-notify',
                    builder: (context, state) => const SizedBox(),
                  ),
                ],
              ),
            ],
          ),
          // Branch 4: 我的
          StatefulShellBranch(
            routes: [
              GoRoute(
                path: AppRouterConfig.profile,
                name: 'profile',
                builder: (context, state) => const ProfilePage(),
                routes: [
                  GoRoute(
                    path: 'task-history',
                    name: 'task-history',
                    builder: (context, state) => const TaskHistoryPage(),
                  ),
                  GoRoute(
                    path: 'files',
                    name: 'files',
                    builder: (context, state) => const FilesPage(),
                  ),
                  GoRoute(
                    path: 'orders',
                    name: 'orders',
                    builder: (context, state) => const OrdersPage(),
                  ),
                  GoRoute(
                    path: 'quota',
                    name: 'quota',
                    builder: (context, state) => const QuotaPage(),
                  ),
                  GoRoute(
                    path: 'member',
                    name: 'member',
                    builder: (context, state) => const MemberPage(),
                  ),
                  GoRoute(
                    path: 'package',
                    name: 'package',
                    builder: (context, state) => const PackagePage(),
                  ),
                  GoRoute(
                    path: 'feedback',
                    name: 'feedback',
                    builder: (context, state) => const FeedbackPage(),
                  ),
                  GoRoute(
                    path: 'favorites',
                    name: 'favorites',
                    builder: (context, state) => const FavoritesPage(),
                  ),
                  GoRoute(
                    path: 'settings',
                    name: 'settings',
                    builder: (context, state) => const SettingsPage(),
                  ),
                  GoRoute(
                    path: 'help',
                    name: 'help',
                    builder: (context, state) => const HelpPage(),
                  ),
                  GoRoute(
                    path: 'about',
                    name: 'about',
                    builder: (context, state) => const AboutPage(),
                  ),
                  GoRoute(
                    path: 'notify',
                    name: 'notify',
                    builder: (context, state) => const NotifyPage(),
                  ),
                  // ==================== 管理模块 (L2, /profile/...) ====================
                  GoRoute(
                    path: 'dashboard',
                    name: 'dashboard',
                    builder: (context, state) => const DashboardPage(),
                  ),
                  GoRoute(
                    path: 'system/user-manage',
                    name: 'user-manage',
                    builder: (context, state) => const UserManagePage(),
                  ),
                  GoRoute(
                    path: 'system/role-manage',
                    name: 'role-manage',
                    builder: (context, state) => const RoleManagePage(),
                  ),
                  GoRoute(
                    path: 'system/menu-manage',
                    name: 'menu-manage',
                    builder: (context, state) => const MenuManagePage(),
                  ),
                  GoRoute(
                    path: 'system/dept-manage',
                    name: 'dept-manage',
                    builder: (context, state) => const DeptManagePage(),
                  ),
                  GoRoute(
                    path: 'system/dict-manage',
                    name: 'dict-manage',
                    builder: (context, state) => const DictManagePage(),
                  ),
                  GoRoute(
                    path: 'system/algorithm-manage',
                    name: 'algorithm-manage',
                    builder: (context, state) => const AlgorithmManagePage(),
                  ),
                  GoRoute(
                    path: 'system/dataset-manage',
                    name: 'dataset-manage',
                    builder: (context, state) => const DatasetManagePage(),
                  ),
                  GoRoute(
                    path: 'system/task-manage',
                    name: 'task-manage',
                    builder: (context, state) => const TaskManagePage(),
                  ),
                  GoRoute(
                    path: 'system/member-manage',
                    name: 'member-manage',
                    builder: (context, state) => const MemberManagePage(),
                  ),
                  GoRoute(
                    path: 'system/package-manage',
                    name: 'package-manage',
                    builder: (context, state) => const PackageManagePage(),
                  ),
                  GoRoute(
                    path: 'system/order-manage',
                    name: 'order-manage',
                    builder: (context, state) => const OrderManagePage(),
                  ),
                  GoRoute(
                    path: 'system/feedback-manage',
                    name: 'feedback-manage',
                    builder: (context, state) => const FeedbackManagePage(),
                  ),
                  GoRoute(
                    path: 'system/message-manage',
                    name: 'message-manage',
                    builder: (context, state) => const MessageManagePage(),
                  ),
                  GoRoute(
                    path: 'system/recommend-manage',
                    name: 'recommend-manage',
                    builder: (context, state) => const RecommendManagePage(),
                  ),
                ],
              ),
            ],
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

      if (!isLoggedIn && !isPublicRoute && !isGoingToAuthPage) {
        return AppRouterConfig.login;
      }

      if (isLoggedIn && isGoingToAuthPage) {
        return AppRouterConfig.home;
      }

      return null;
    },
  );
});
