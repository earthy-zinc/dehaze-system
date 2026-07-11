import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'core/auth/auth_error_handler.dart';
import 'providers/auth_provider.dart';
import 'providers/providers.dart';
import 'router/config.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 初始化 SharedPreferences
  final sharedPreferences = await SharedPreferences.getInstance();

  runApp(
    ProviderScope(
      overrides: [
        sharedPreferencesProvider.overrideWithValue(sharedPreferences),
      ],
      child: const DehazeApp(),
    ),
  );
}

/// 应用根 Widget
class DehazeApp extends ConsumerStatefulWidget {
  const DehazeApp({super.key});

  @override
  ConsumerState<DehazeApp> createState() => _DehazeAppState();
}

class _DehazeAppState extends ConsumerState<DehazeApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    // 设置认证错误回调（Token 过期且刷新失败时触发）
    AuthErrorHandler.setHandler(() {
      ref.read(authProvider.notifier).onAuthError();
    });

    // 初始化认证状态
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(authProvider.notifier).initialize();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(goRouterProvider);

    return MaterialApp.router(
      title: '图像去雾应用',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.light,
      routerConfig: router,
    );
  }
}
