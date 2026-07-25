import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/network/api_result.dart';
import '../core/storage/token_storage.dart';
import '../models/auth_model.dart';
import '../models/user_model.dart';
import '../providers/providers.dart';
import '../services/auth_service.dart';

class AuthState {
  const AuthState({
    this.user,
    this.sessionId,
    this.status = AuthStatus.initial,
    this.errorMessage,
  });

  final UserModel? user;
  final String? sessionId;
  final AuthStatus status;
  final String? errorMessage;

  bool get isAuthenticated => user != null && sessionId != null;
  bool get isLoading => status == AuthStatus.loading;

  AuthState copyWith({
    UserModel? user,
    String? sessionId,
    AuthStatus? status,
    String? errorMessage,
    bool clearUser = false,
    bool clearSessionId = false,
  }) =>
      AuthState(
        user: clearUser ? null : (user ?? this.user),
        sessionId: clearSessionId ? null : (sessionId ?? this.sessionId),
        status: status ?? this.status,
        errorMessage: errorMessage,
      );
}

enum AuthStatus {
  initial,
  loading,
  authenticated,
  unauthenticated,
  error,
}

class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier(this._authService, this._tokenStorage)
      : super(const AuthState());

  final AuthService _authService;
  final TokenStorage _tokenStorage;

  Future<void> initialize() async {
    if (_tokenStorage.hasToken) {
      state = state.copyWith(
        sessionId: _tokenStorage.sessionId,
        status: AuthStatus.loading,
      );
      try {
        final user = await _authService.getCurrentUser();
        state = state.copyWith(
          user: user,
          status: AuthStatus.authenticated,
        );
      } catch (_) {
        await _tokenStorage.clearTokens();
        state = const AuthState(status: AuthStatus.unauthenticated);
      }
    } else {
      state = const AuthState(status: AuthStatus.unauthenticated);
    }
  }

  Future<void> login(LoginRequest request) async {
    state = state.copyWith(status: AuthStatus.loading, errorMessage: null);

    try {
      final response = await _authService.login(request);

      await _tokenStorage.saveSessionId(response.sessionId);

      final user = await _authService.getCurrentUser();

      state = AuthState(
        user: user,
        sessionId: response.sessionId,
        status: AuthStatus.authenticated,
      );
    } catch (e) {
      state = AuthState(
        status: AuthStatus.error,
        errorMessage: extractErrorMessage(e),
      );
    }
  }

  Future<void> logout() async {
    try {
      await _authService.logout();
    } catch (_) {
    }

    await _tokenStorage.clearTokens();
    state = const AuthState(status: AuthStatus.unauthenticated);
  }

  Future<void> onAuthError() async {
    await _tokenStorage.clearTokens();
    state = const AuthState(status: AuthStatus.unauthenticated);
  }

  void clearError() {
    if (state.status == AuthStatus.error) {
      state = state.copyWith(status: AuthStatus.unauthenticated);
    }
  }
}

final authServiceProvider = Provider<AuthService>((ref) {
  final dio = ref.watch(dioClientProvider);
  return AuthService(dio);
});

final authProvider = StateNotifierProvider<AuthNotifier, AuthState>((ref) {
  final authService = ref.watch(authServiceProvider);
  final tokenStorage = ref.watch(tokenStorageProvider);
  return AuthNotifier(authService, tokenStorage);
});
