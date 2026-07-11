/// 认证错误处理容器
///
/// 用于打破 Provider 循环依赖：
/// dioClientProvider → authErrorCallback → authProvider → authServiceProvider → dioClientProvider
///
/// 通过静态持有回调函数，避免在 Provider 层级中直接引用 authProvider
class AuthErrorHandler {
  AuthErrorHandler._();

  /// 认证失败回调（在 DehazeApp 中设置）
  static void Function()? _onAuthError;

  /// 设置认证失败回调
  static void setHandler(void Function()? handler) {
    _onAuthError = handler;
  }

  /// 触发认证失败回调
  static void handle() {
    _onAuthError?.call();
  }
}
