import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../models/auth_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/providers.dart';
import '../../router/config.dart';
import '../../theme/app_theme.dart';
import '../../utils/ui_utils.dart';

class RegisterPage extends ConsumerStatefulWidget {
  const RegisterPage({super.key});

  @override
  ConsumerState<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends ConsumerState<RegisterPage> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _nicknameController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _captchaController = TextEditingController();

  CaptchaResponse? _captcha;
  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;
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
    _nicknameController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _captchaController.dispose();
    super.dispose();
  }

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
        showError(context, '获取验证码失败，请检查网络连接');
      }
    }
  }

  Future<void> _register() async {
    if (!_formKey.currentState!.validate()) return;
    if (_captcha == null) {
      showError(context, '请先获取验证码');
      return;
    }

    await ref.read(authProvider.notifier).register(
          username: _usernameController.text.trim(),
          password: _passwordController.text,
          nickname: _nicknameController.text.trim(),
          captchaKey: _captcha!.captchaKey,
          captchaCode: _captchaController.text.trim(),
        );

    if (!mounted) return;

    final authState = ref.read(authProvider);
    if (authState.status == AuthStatus.authenticated) {
      context.go(AppRouterConfig.home);
    } else if (authState.status == AuthStatus.error) {
      showError(context, authState.errorMessage ?? '注册失败');
      _captchaController.clear();
      _loadCaptcha();
    }
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authProvider);
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.surface,
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
                    _buildHeader(theme),
                    const SizedBox(height: 36),
                    _buildInputField(
                      controller: _usernameController,
                      label: '用户名',
                      icon: Icons.person_outline,
                      hint: '请输入用户名',
                      validator: (v) => (v == null || v.trim().isEmpty) ? '请输入用户名' : null,
                    ),
                    const SizedBox(height: 16),
                    _buildInputField(
                      controller: _nicknameController,
                      label: '昵称',
                      icon: Icons.badge_outlined,
                      hint: '请输入昵称',
                      validator: (v) => (v == null || v.trim().isEmpty) ? '请输入昵称' : null,
                    ),
                    const SizedBox(height: 16),
                    _buildInputField(
                      controller: _passwordController,
                      label: '密码',
                      icon: Icons.lock_outline,
                      hint: '请输入密码',
                      obscureText: _obscurePassword,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscurePassword
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                        ),
                        onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                      ),
                      validator: (v) => (v == null || v.isEmpty) ? '请输入密码' : null,
                    ),
                    const SizedBox(height: 16),
                    _buildInputField(
                      controller: _confirmPasswordController,
                      label: '确认密码',
                      icon: Icons.lock_outline,
                      hint: '请再次输入密码',
                      obscureText: _obscureConfirmPassword,
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscureConfirmPassword
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                        ),
                        onPressed: () => setState(() => _obscureConfirmPassword = !_obscureConfirmPassword),
                      ),
                      validator: (v) {
                        if (v == null || v.isEmpty) return '请再次输入密码';
                        if (v != _passwordController.text) return '两次密码不一致';
                        return null;
                      },
                    ),
                    const SizedBox(height: 16),
                    _buildCaptchaRow(theme),
                    const SizedBox(height: 28),
                    FilledButton(
                      onPressed: authState.isLoading ? null : _register,
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
                          : const Text('注册', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '已有账号？',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        TextButton(
                          onPressed: () => context.go(AppRouterConfig.login),
                          style: TextButton.styleFrom(
                            padding: EdgeInsets.zero,
                            minimumSize: Size.zero,
                            tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          ),
                          child: const Text('立即登录'),
                        ),
                      ],
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

  Widget _buildInputField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    required String hint,
    required String? Function(String?) validator,
    TextInputAction textInputAction = TextInputAction.next,
    bool obscureText = false,
    Widget? suffixIcon,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(
            label,
            style: Theme.of(context).textTheme.labelLarge?.copyWith(
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        TextFormField(
          controller: controller,
          obscureText: obscureText,
          decoration: InputDecoration(
            hintText: hint,
            prefixIcon: Icon(icon, size: 20),
            suffixIcon: suffixIcon,
          ),
          validator: validator,
          textInputAction: textInputAction,
        ),
      ],
    );
  }

  Widget _buildCaptchaRow(ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 8),
          child: Text(
            '验证码',
            style: theme.textTheme.labelLarge?.copyWith(
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: TextFormField(
                controller: _captchaController,
                decoration: const InputDecoration(
                  hintText: '请输入验证码',
                  prefixIcon: Icon(Icons.shield_outlined, size: 20),
                ),
                validator: (v) => (v == null || v.isEmpty) ? '请输入验证码' : null,
                textInputAction: TextInputAction.done,
              ),
            ),
            const SizedBox(width: 12),
            GestureDetector(
              onTap: _isCaptchaLoading ? null : _loadCaptcha,
              child: Container(
                width: 140,
                height: 56,
                decoration: BoxDecoration(
                  border: Border.all(color: theme.colorScheme.outline),
                  borderRadius: BorderRadius.circular(AppTheme.radiusM),
                ),
                child: _isCaptchaLoading || _captcha == null
                    ? Center(
                        child: _isCaptchaLoading
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.refresh, size: 18, color: theme.colorScheme.onSurfaceVariant),
                                  SizedBox(height: 2),
                                  Text('点击获取', style: TextStyle(fontSize: 11, color: theme.colorScheme.onSurfaceVariant)),
                                ],
                              ),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(AppTheme.radiusM),
                        child: Image.memory(
                          _decodeBase64Image(_captcha!.captchaBase64),
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
      ],
    );
  }

  Widget _buildHeader(ThemeData theme) => Column(
    children: [
      Container(
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          gradient: AppTheme.getPrimaryGradient(),
          borderRadius: BorderRadius.circular(22),
          boxShadow: [
            BoxShadow(
              color: AppTheme.brandBlue.withValues(alpha: 0.35),
              blurRadius: 24,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: const Icon(Icons.cloud_outlined, color: Colors.white, size: 42),
      ),
      const SizedBox(height: 20),
      Text(
        '创建新账号',
        style: theme.textTheme.headlineSmall?.copyWith(
          fontWeight: FontWeight.w700,
        ),
      ),
      const SizedBox(height: 8),
      Text(
        '注册以使用图像去雾系统',
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
    ],
  );

  static Uint8List _decodeBase64Image(String base64Str) {
    final pureBase64 = base64Str.contains(',')
        ? base64Str.split(',').last
        : base64Str;
    return base64Decode(pureBase64);
  }
}
