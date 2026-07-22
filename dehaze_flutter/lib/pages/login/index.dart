import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/auth_model.dart';
import '../../providers/auth_provider.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';

/// 登录页面
class LoginPage extends ConsumerStatefulWidget {
  const LoginPage({super.key});

  @override
  ConsumerState<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends ConsumerState<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  final _captchaController = TextEditingController();

  CaptchaResponse? _captcha;
  bool _obscurePassword = true;
  bool _isCaptchaLoading = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadCaptcha();
    });
  }

  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    _captchaController.dispose();
    super.dispose();
  }

  /// 加载验证码
  Future<void> _loadCaptcha() async {
    setState(() => _isCaptchaLoading = true);
    try {
      final authService = ref.read(authServiceProvider);
      final captcha = await authService.getCaptcha();
      if (mounted) {
        setState(() {
          _captcha = captcha;
          _isCaptchaLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isCaptchaLoading = false);
        _showError('获取验证码失败，请检查网络连接');
      }
    }
  }

  /// 执行登录
  Future<void> _login() async {
    if (!_formKey.currentState!.validate()) return;
    if (_captcha == null) {
      _showError('请先获取验证码');
      return;
    }

    final request = LoginRequest(
      username: _usernameController.text.trim(),
      password: _passwordController.text,
      captchaKey: _captcha!.captchaKey,
      captchaCode: _captchaController.text.trim(),
    );

    await ref.read(authProvider.notifier).login(request);

    if (!mounted) return;

    final authState = ref.read(authProvider);
    if (authState.status == AuthStatus.authenticated) {
      context.go(AppRouterConfig.home);
    } else if (authState.status == AuthStatus.error) {
      _showError(authState.errorMessage ?? '登录失败');
      // 刷新验证码
      _captchaController.clear();
      _loadCaptcha();
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Theme.of(context).colorScheme.error,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final theme = Theme.of(context);

    return Scaffold(
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 400),
              child: Form(
                key: _formKey,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Logo 和标题
                    _buildHeader(theme),

                    const SizedBox(height: 32),

                    // 用户名
                    TextFormField(
                      controller: _usernameController,
                      decoration: const InputDecoration(
                        labelText: '用户名',
                        prefixIcon: Icon(Icons.person_outline),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '请输入用户名';
                        }
                        return null;
                      },
                      textInputAction: TextInputAction.next,
                    ),

                    const SizedBox(height: 16),

                    // 密码
                    TextFormField(
                      controller: _passwordController,
                      decoration: InputDecoration(
                        labelText: '密码',
                        prefixIcon: const Icon(Icons.lock_outline),
                        suffixIcon: IconButton(
                          icon: Icon(
                            _obscurePassword
                                ? Icons.visibility_off_outlined
                                : Icons.visibility_outlined,
                          ),
                          onPressed: () {
                            setState(() {
                              _obscurePassword = !_obscurePassword;
                            });
                          },
                        ),
                      ),
                      obscureText: _obscurePassword,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '请输入密码';
                        }
                        return null;
                      },
                      textInputAction: TextInputAction.next,
                    ),

                    const SizedBox(height: 16),

                    // 验证码
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: TextFormField(
                            controller: _captchaController,
                            decoration: const InputDecoration(
                              labelText: '验证码',
                              prefixIcon: Icon(Icons.shield_outlined),
                            ),
                            validator: (value) {
                              if (value == null || value.isEmpty) {
                                return '请输入验证码';
                              }
                              return null;
                            },
                            textInputAction: TextInputAction.done,
                          ),
                        ),
                        const SizedBox(width: 12),
                        // 验证码图片
                        GestureDetector(
                          onTap: _isCaptchaLoading ? null : _loadCaptcha,
                          child: Container(
                            width: 160,
                            height: 56,
                            decoration: BoxDecoration(
                              border: Border.all(
                                color: theme.colorScheme.outline,
                              ),
                              borderRadius: BorderRadius.circular(
                                AppTheme.radiusM,
                              ),
                            ),
                            child: _isCaptchaLoading || _captcha == null
                                ? Center(
                                    child: _isCaptchaLoading
                                        ? const SizedBox(
                                            width: 20,
                                            height: 20,
                                            child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                            ),
                                          )
                                        : const Icon(Icons.refresh),
                                  )
                                : ClipRRect(
                                    borderRadius: BorderRadius.circular(
                                      AppTheme.radiusM,
                                    ),
                                    child: Image.memory(
                                      _decodeBase64Image(
                                        _captcha!.captchaBase64,
                                      ),
                                      // fill 保证完整显示所有验证码字符（cover 会裁剪）
                                      fit: BoxFit.fill,
                                      gaplessPlayback: true,
                                      errorBuilder: (_, _, _) => const Center(
                                        child: Icon(Icons.broken_image),
                                      ),
                                    ),
                                  ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 8),

                    // 刷新验证码提示
                    Align(
                      alignment: Alignment.centerRight,
                      child: TextButton.icon(
                        onPressed: _isCaptchaLoading ? null : _loadCaptcha,
                        icon: const Icon(Icons.refresh, size: 16),
                        label: const Text('刷新验证码'),
                        style: TextButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          minimumSize: const Size(0, 32),
                          textStyle: const TextStyle(fontSize: 12),
                        ),
                      ),
                    ),

                    const SizedBox(height: 24),

                    // 登录按钮
                    FilledButton(
                      onPressed: authState.isLoading ? null : _login,
                      style: FilledButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(AppTheme.radiusM),
                        ),
                      ),
                      child: authState.isLoading
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Text('登录', style: TextStyle(fontSize: 16)),
                    ),

                    const SizedBox(height: 16),

                    // 返回首页
                    TextButton(
                      onPressed: () => context.go(AppRouterConfig.home),
                      child: const Text('返回首页'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  /// 构建头部 Logo
  Widget _buildHeader(ThemeData theme) => Column(
    children: [
      Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          gradient: AppTheme.getPrimaryGradient(),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: AppTheme.brandBlue.withValues(alpha: 0.3),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: const Icon(Icons.cloud_outlined, color: Colors.white, size: 40),
      ),
      const SizedBox(height: 16),
      Text(
        '图像去雾系统',
        style: theme.textTheme.headlineSmall?.copyWith(
          fontWeight: FontWeight.w700,
        ),
      ),
      const SizedBox(height: 8),
      Text(
        '请登录以使用完整功能',
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
    ],
  );

  /// 解码 Base64 验证码图片
  static Uint8List _decodeBase64Image(String base64Str) {
    // 移除可能的前缀（如 data:image/png;base64,）
    final pureBase64 = base64Str.contains(',')
        ? base64Str.split(',').last
        : base64Str;
    return base64Decode(pureBase64);
  }
}
