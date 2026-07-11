import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/auth/auth_error_handler.dart';
import '../core/network/api_client.dart';
import '../core/storage/token_storage.dart';

// ==================== 基础设施 Providers ====================

/// SharedPreferences Provider（必须在 main.dart 中 override）
final sharedPreferencesProvider = Provider<SharedPreferences>((ref) {
  throw UnimplementedError(
    'SharedPreferences must be initialized in main.dart',
  );
});

/// Token 存储 Provider
final tokenStorageProvider = Provider<TokenStorage>((ref) {
  final prefs = ref.watch(sharedPreferencesProvider);
  return TokenStorage(prefs);
});

/// 认证错误回调 Provider
///
/// 使用 AuthErrorHandler 静态容器，避免 Provider 循环依赖。
/// 在 DehazeApp 初始化时通过 AuthErrorHandler.setHandler 设置实际回调。
final authErrorCallbackProvider = Provider<void Function()>((ref) {
  return AuthErrorHandler.handle;
});

/// Dio Provider
final dioClientProvider = Provider<Dio>((ref) {
  final tokenStorage = ref.watch(tokenStorageProvider);
  final onAuthError = ref.watch(authErrorCallbackProvider);
  final apiClient = ApiClient.create(
    tokenStorage: tokenStorage,
    onAuthError: onAuthError,
  );
  return apiClient.dio;
});

/// ApiClient Provider
final apiClientProvider = Provider<ApiClient>((ref) {
  final tokenStorage = ref.watch(tokenStorageProvider);
  final onAuthError = ref.watch(authErrorCallbackProvider);
  return ApiClient.create(
    tokenStorage: tokenStorage,
    onAuthError: onAuthError,
  );
});
