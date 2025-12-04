import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api/api_service.dart';
import 'providers/providers.dart';
import 'router/config.dart';
import 'theme/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 初始化SharedPreferences
  final sharedPreferences = await SharedPreferences.getInstance();

  final apiService = APIService();

  // 初始化APIService
  apiService.initialize();

  runApp(
    ProviderScope(
      overrides: [
        sharedPreferencesProvider.overrideWithValue(sharedPreferences),
        dioClientProvider.overrideWithValue(apiService.dio),
      ],
      child: MaterialApp.router(
          title: '图像去雾应用',
          debugShowCheckedModeBanner: false,
          theme: AppTheme.lightTheme,
          darkTheme: AppTheme.darkTheme,
          themeMode: ThemeMode.light,
          routerConfig: AppRouterConfig.router,
        ),
    ),
  );
}
