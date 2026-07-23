import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/network/api_result.dart';
import '../core/storage/token_storage.dart';
import '../models/auth_model.dart';
import '../models/user_model.dart';
import '../providers/providers.dart';
import '../services/auth_service.dart';

/// 认证状态
class AuthState {
  const AuthState({
    this.user,
    this.token,
    this.status = AuthStatus.initial,
    this.errorMessage,
  });

  final UserModel? user;
  final String? token;
  final AuthStatus status;
  final String? errorMessage;

  bool get isAuthenticated => user != null && token != null;
  bool get isLoading => status == AuthStatus.loading;

  AuthState copyWith({
    UserModel? user,
    String? token,
    AuthStatus? status,
    String? errorMessage,
    bool clearUser = false,
    bool clearToken = false,
  }) =>
      AuthState(
        user: clearUser ? null : (user ?? this.user),
        token: clearToken ? null : (token ?? this.token),
        status: status ?? this.status,
        errorMessage: errorMessage,
      );
}

/// 认证状态枚举
enum AuthStatus {
  initial,
  loading,
  authenticated,
  unauthenticated,
  error,
}

/// 认证状态管理
class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._authService, this._tokenStorage)
      : super(const AuthState());

  final AuthService _authService;
  final TokenStorage _tokenStorage;

  /// 初始化：从本地存储恢复登录态
  Future<void> initialize() async {
    if (_tokenStorage.hasToken) {
      state = state.copyWith(
        token: _tokenStorage.accessToken,
        status: AuthStatus.loading,
      );
      try {
        final user = await _authService.getCurrentUser();
        state = state.copyWith(
          user: user,
          status: AuthStatus.authenticated,
        );
      } catch (_) {
        // Token 已失效，清除
        await _tokenStorage.clearTokens();
        state = const AuthState(status: AuthStatus.unauthenticated);
      }
    } else {
      state = const AuthState(status: AuthStatus.unauthenticated);
    }
  }

  /// 登录
  Future<void> login(LoginRequest request) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);

    try {
      final response = await _authService.login(request);

      // 保存 Token
      await _tokenStorage.saveTokens(
        accessToken: response.accessToken,
        refreshToken: response.refreshToken,
      );

      // 获取用户信息
      final user = await _authService.getCurrentUser();

      state = AuthState(
        user: user,
        token: response.accessToken,
        status: AuthStatus.authenticated,
      );
    } catch (e) {
      state = AuthState(
        status: AuthStatus.error,
        errorMessage: extractErrorMessage(e),
      );
    }
  }

  /// 登出
  Future<void> logout() async {
    try {
      await _authService.logout();
    } catch (_) {
      // 即使登出 API 失败，也清除本地状态
    }

    await _tokenStorage.clearTokens();
    state = const AuthState(status: AuthStatus.unauthenticated);
  }

  /// 认证失败时被调用（由拦截器触发）
  Future<void> onAuthError() async {
    await _tokenStorage.clearTokens();
    state = const AuthState(status: AuthStatus.unauthenticated);
  }

  /// 清除错误状态
  void clearError() {
    if (state.status == AuthStatus.error) {
      state = state.copyWith(status: AuthStatus.unauthenticated);
    }
  }
}

// ==================== Providers ====================

/// 认证服务 Provider
final authServiceProvider = Provider<AuthService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return AuthService(dio);
});

/// 认证状态 Provider
final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  final authService = ref.watch(authServiceProvider);
  final tokenStorage = ref.watch(tokenStorageProvider);
  return AuthNotifier(authService, tokenStorage);
});
