import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/member_model.dart';
import '../../providers/providers.dart';
import '../../theme/app_theme.dart';

/// 我的会员 — L2 页面
///
/// 展示会员等级、成长值、权益，对接 MemberService
class MemberPage extends ConsumerStatefulWidget {
  const MemberPage({super.key});

  @override
  ConsumerState<MemberPage> createState() => _MemberPageState();
}

class _MemberPageState extends ConsumerState<MemberPage> {
  bool _isLoading = true;
  String? _error;

  MemberProfileVO? _profile;
  List<BenefitVO> _benefits = [];
  SignInCalendarVO? _calendar;
  bool _isSigningIn = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _load());
  }

  Future<void> _load() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final memberService = ref.read(memberServiceProvider);
      final now = DateTime.now();
      final results = await Future.wait([
        memberService.getProfile(),
        memberService.getBenefits(),
        memberService.getSignInCalendar(year: now.year, month: now.month),
      ]);
      if (!mounted) return;
      setState(() {
        _profile = results[0] as MemberProfileVO;
        _benefits = results[1] as List<BenefitVO>;
        _calendar = results[2] as SignInCalendarVO;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _handleSignIn() async {
    setState(() => _isSigningIn = true);
    try {
      final result = await ref.read(memberServiceProvider).signIn();
      if (!mounted) return;
      _showSignInResult(result);
      // 签到成功后刷新数据
      await _load();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('签到失败: $e')));
    } finally {
      if (mounted) setState(() => _isSigningIn = false);
    }
  }

  void _showSignInResult(SignInResultVO result) {
    showDialog<void>(
      context: context,
      builder:
          (_) => AlertDialog(
            title: const Text('签到成功'),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('获得成长值: +${result.growthValue}'),
                if (result.bonusGrowth > 0)
                  Text('连续签到奖励: +${result.bonusGrowth}'),
                Text('已连续签到 ${result.continuousDays} 天'),
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('确定'),
              ),
            ],
          ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('我的会员')),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _buildError(theme)
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _buildLevelCard(theme),
                      const SizedBox(height: 20),
                      _buildSignInSection(theme),
                      const SizedBox(height: 20),
                      Text(
                        '会员权益',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 12),
                      ..._benefits.map(
                        (b) => Card(
                          margin: const EdgeInsets.only(bottom: 8),
                          child: ListTile(
                            leading: Icon(
                              _benefitIcon(b.levelCode),
                              color: AppTheme.brandBlue,
                            ),
                            title: Text(b.levelName),
                            subtitle: Text(
                              '去雾配额: ${b.monthlyDehazeQuota}次/月 | 评估配额: ${b.monthlyEvaluateQuota}次/月',
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }

  Widget _buildLevelCard(ThemeData theme) {
    final profile = _profile;
    final levelName = profile?.levelName ?? '普通用户';
    final growthValue = profile?.growthValue ?? 0;
    final nextLevelGrowth = profile?.nextLevelGrowth;
    final progressPercent = profile?.progressPercent ?? 0;

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFFFFD700), Color(0xFFFFA500)],
        ),
        borderRadius: BorderRadius.circular(AppTheme.radiusL),
      ),
      child: Column(
        children: [
          const Icon(Icons.workspace_premium, color: Colors.white, size: 48),
          const SizedBox(height: 12),
          Text(
            levelName,
            style: theme.textTheme.headlineSmall?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '成长值: $growthValue${nextLevelGrowth != null ? ' / $nextLevelGrowth' : ''}',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: Colors.white.withValues(alpha: 0.9),
            ),
          ),
          if (nextLevelGrowth != null) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: progressPercent / 100,
                minHeight: 6,
                backgroundColor: Colors.white.withValues(alpha: 0.3),
                valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '距下一级还需 ${nextLevelGrowth - growthValue} 成长值',
              style: theme.textTheme.bodySmall?.copyWith(
                color: Colors.white.withValues(alpha: 0.8),
              ),
            ),
          ],
          const SizedBox(height: 16),
          FilledButton(
            onPressed: () {},
            style: FilledButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: AppTheme.brandBlue,
            ),
            child: const Text('立即升级'),
          ),
        ],
      ),
    );
  }

  Widget _buildSignInSection(ThemeData theme) {
    final calendar = _calendar;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '每日签到',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                FilledButton.icon(
                  onPressed: _isSigningIn ? null : _handleSignIn,
                  icon: _isSigningIn
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.check, size: 18),
                  label: Text(_isSigningIn ? '签到中...' : '签到'),
                ),
              ],
            ),
            if (calendar != null) ...[
              const SizedBox(height: 8),
              Text(
                '本月已签到 ${calendar.totalDays} 天 · 连续 ${calendar.continuousDays} 天',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  IconData _benefitIcon(String levelCode) {
    switch (levelCode) {
      case 'level_0':
        return Icons.person_outline;
      case 'level_1':
        return Icons.star_outline;
      case 'level_2':
        return Icons.star_half;
      case 'level_3':
        return Icons.star;
      default:
        return Icons.card_giftcard;
    }
  }

  Widget _buildError(ThemeData theme) => Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 48, color: theme.colorScheme.error),
            const SizedBox(height: 12),
            Text(_error!, style: theme.textTheme.bodyMedium),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _load, child: const Text('重试')),
          ],
        ),
      );
}
