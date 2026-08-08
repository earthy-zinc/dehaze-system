import 'dart:async';
import 'dart:ui';

import 'package:flutter/foundation.dart' show debugPrint, kDebugMode;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'constants/app_constants.dart';
import 'core/auth/auth_error_handler.dart';
import 'core/logger/logger.dart';
import 'providers/auth_provider.dart';
import 'providers/providers.dart';
import 'router/config.dart';
import 'theme/app_theme.dart';

void main() {
  // 必须在 runZonedGuarded 内部初始化 binding 与调用 runApp，否则 Flutter 会抛 Zone mismatch
  // assertion（绑定初始化 zone 与运行 zone 不一致，导致 zone-specific 配置失效）。
  runZonedGuarded(
    () async {
      WidgetsFlutterBinding.ensureInitialized();

      // 初始化 SharedPreferences
      final sharedPreferences = await SharedPreferences.getInstance();

      // 初始化日志 Logger（崩溃捕获依赖，须在 runApp 前完成）
      final logger = Logger.init(
        app: 'flutter',
        appVersion: AppConstants.appVersion,
      );

      // 生产环境启动补报：从本地文件读取崩溃遗留日志并上报（§3.5）
      if (!kDebugMode) {
        unawaited(logger.flushFromDisk());
      }

      // Flutter 框架错误捕获（含布局/渲染错误）
      FlutterError.onError = (details) {
        // 不调用 FlutterError.presentError：它默认调用 dumpErrorToConsole 会输出
        // `══╡ EXCEPTION CAUGHT BY ... ╞══` 或 `Another exception was thrown:`，
        // 与 Logger 的 ConsoleTransport 输出重复。Logger 的 error_stack 使用
        // details.toString()（与 dumpErrorToConsole 同源），信息完整不丢失
        // error_stack 含 library 来源、widget 上下文（Creator 链）、RenderObject 诊断、
        // 约束等完整诊断。仅用 details.exception + 纯 stack 无法定位布局错误源 widget
        Logger.instance.error(
          'FlutterError${details.library != null ? ' [${details.library}]' : ''}: ${details.exceptionAsString()}',
          errorType: 'dart',
          errorSource: 'FlutterError',
          errorStack: details.toString(),
        );
      };

      // 平台消息错误捕获
      PlatformDispatcher.instance.onError = (error, stack) {
        Logger.instance.error(
          'PlatformDispatcherError: $error',
          errorType: 'dart',
          errorSource: 'PlatformDispatcher',
          errorStack: stack.toString(),
        );
        return true;
      };

      runApp(
        ProviderScope(
          overrides: [
            sharedPreferencesProvider.overrideWithValue(sharedPreferences),
          ],
          child: const DehazeApp(),
        ),
      );
    },
    // Zone 未处理异步异常（Logger 可能尚未初始化，需降级避免二次崩溃）
    (error, stackTrace) {
      if (Logger.isInitialized) {
        Logger.instance.error(
          'Uncaught zone error: $error',
          errorType: 'dart',
          errorSource: 'Zone',
          errorStack: stackTrace.toString(),
        );
      } else {
        debugPrint(
          '[dehaze][ERROR] Uncaught zone error before Logger init: $error\n$stackTrace',
        );
      }
    },
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
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // App 进入后台时补报日志
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      Logger.instance.flushOnBackground();
    }
  }

  @override
  Widget build(BuildContext context) {
    final router = ref.watch(goRouterProvider);
    // 注入 router，供 Logger 在生成日志时自动填充 url 字段（ELK 按页面过滤）
    Logger.instance.attachRouter(router);

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
