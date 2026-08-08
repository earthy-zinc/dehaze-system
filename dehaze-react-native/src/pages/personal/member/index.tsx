/**
 * 我的会员 (L2)
 *
 * MemberAPI.getProfile 等级/成长值/权益
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { MemberAPI } from 'dehaze-sdk-js';
import type { MemberProfileVO } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';
import LinearGradient from 'react-native-linear-gradient';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const LEVEL_COLORS: Record<string, string[]> = {
  level_0: ['#9ca3af', '#d1d5db'],
  level_1: ['#9ca3af', '#d1d5db'],
  level_2: ['#f59e0b', '#d97706'],
  level_3: ['#8b5cf6', '#6d28d9'],
};

const PersonalMemberScreen: React.FC = () => {
  const navigation = useNavigation();
  const [member, setMember] = useState<MemberProfileVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadMember = useCallback(async () => {
    try {
      setError(null);
      const profile = await MemberAPI.getProfile();
      setMember(profile);
    } catch {
      setMember(null);
      setError('获取会员信息失败，请重试');
      Alert.alert('加载失败', '获取会员信息失败，请重试');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMember();
  }, [loadMember]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadMember();
    setRefreshing(false);
  }, [loadMember]);

  const colors = LEVEL_COLORS[member?.levelCode || 'level_0'] || LEVEL_COLORS.level_0;
  const isVip = member && member.levelCode !== 'level_0' && member.levelCode !== 'level_1';
  const nextGrowth = member?.nextLevelGrowth;
  const growthProgress = nextGrowth && nextGrowth > 0
    ? Math.min(1, (member?.growthValue || 0) / nextGrowth)
    : 0;

  const benefits = member?.benefits;
  const benefitList: { label: string; value: string }[] = benefits
    ? [
        { label: '月度去雾额度', value: `${benefits.monthlyDehazeQuota} 次` },
        { label: '月度评估额度', value: `${benefits.monthlyEvaluateQuota} 次` },
        { label: '批量处理上限', value: `${benefits.batchLimit} 张` },
        { label: '历史保留', value: `${benefits.historyRetention} 天` },
        { label: '高级参数', value: benefits.advancedParams ? '支持' : '不支持' },
        { label: '高清导出', value: benefits.hdExport ? '支持' : '不支持' },
        { label: '报告导出', value: benefits.reportExport ? '支持' : '不支持' },
        { label: '批量下载', value: benefits.batchDownload ? '支持' : '不支持' },
      ]
    : [];

  if (loading) {
    return (
      <View style={styles.container}>
        <AppHeader title="会员中心" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>加载中...</Text>
        </View>
      </View>
    );
  }

  if (error && !member) {
    return (
      <View style={styles.container}>
        <AppHeader title="会员中心" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.centered}>
          <Ionicons name="alert-circle-outline" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="会员中心" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
      >
      {/* 会员卡 */}
      <LinearGradient
        colors={colors}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.memberCard}
      >
        <Ionicons name="diamond" size={32} color="#fff" style={styles.cardIcon} />
        <Text style={styles.levelName}>{member?.levelName || '普通用户'}</Text>
        <Text style={styles.growthLabel}>成长值</Text>
        <Text style={styles.growthValue}>{member?.growthValue ?? 0}</Text>
        {nextGrowth ? (
          <>
            <View style={styles.growthBar}>
              <View style={[styles.growthFill, { width: `${growthProgress * 100}%` }]} />
            </View>
            <Text style={styles.growthHint}>距离下一级还需 {nextGrowth - (member?.growthValue || 0)} 成长值</Text>
          </>
        ) : null}
        {member?.expireTime ? (
          <Text style={styles.expireText}>
            会员有效期至 {new Date(member.expireTime).toLocaleDateString('zh-CN')}
          </Text>
        ) : null}
      </LinearGradient>

      {/* 月度用量 */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>本月用量</Text>
        <View style={styles.usageRow}>
          <View style={styles.usageItem}>
            <Text style={styles.usageValue}>{member?.monthlyDehazeUsed ?? 0}/{member?.monthlyDehazeQuota ?? 0}</Text>
            <Text style={styles.usageLabel}>去雾</Text>
          </View>
          <View style={styles.usageItem}>
            <Text style={styles.usageValue}>{member?.monthlyEvaluateUsed ?? 0}/{member?.monthlyEvaluateQuota ?? 0}</Text>
            <Text style={styles.usageLabel}>评估</Text>
          </View>
        </View>
      </View>

      {/* 权益列表 */}
      {benefitList.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>会员权益</Text>
          <View style={styles.benefitCard}>
            {benefitList.map((b, idx) => (
              <View key={b.label} style={[styles.benefitRow, idx < benefitList.length - 1 && styles.benefitDivider]}>
                <Text style={styles.benefitLabel}>{b.label}</Text>
                <Text style={styles.benefitValue}>{b.value}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {!isVip && (
        <View style={styles.upgradeHint}>
          <Ionicons name="diamond-outline" size={18} color="#f59e0b" />
          <Text style={styles.upgradeText}>开通 VIP 会员享受更多权益</Text>
        </View>
      )}
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: theme.spacing.xl },
  loadingText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
  errorText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary, marginTop: theme.spacing.sm, textAlign: 'center' },
  memberCard: {
    borderRadius: theme.layout.borderRadius.xxl,
    padding: theme.spacing.xl,
    alignItems: 'center',
    ...theme.layout.shadows.lg,
  },
  cardIcon: { marginBottom: theme.spacing.sm },
  levelName: {
    fontSize: theme.typography.sizes.h4,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  growthLabel: {
    fontSize: theme.typography.sizes.small,
    color: 'rgba(255,255,255,0.7)',
    marginTop: theme.spacing.md,
  },
  growthValue: {
    fontSize: theme.typography.sizes.h1,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  growthBar: {
    width: '80%',
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.3)',
    borderRadius: 3,
    marginTop: theme.spacing.sm,
    overflow: 'hidden',
  },
  growthFill: {
    height: '100%',
    backgroundColor: '#fff',
    borderRadius: 3,
  },
  growthHint: {
    fontSize: theme.typography.sizes.tiny,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 4,
  },
  expireText: {
    fontSize: theme.typography.sizes.tiny,
    color: 'rgba(255,255,255,0.8)',
    marginTop: theme.spacing.sm,
  },
  section: { marginTop: theme.spacing.lg },
  sectionTitle: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  usageRow: {
    flexDirection: 'row',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    ...theme.layout.shadows.sm,
  },
  usageItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
  },
  usageValue: {
    fontSize: theme.typography.sizes.h6,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  usageLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 4,
  },
  benefitCard: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    ...theme.layout.shadows.sm,
  },
  benefitRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: theme.spacing.md,
  },
  benefitDivider: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.colors.border.light,
  },
  benefitLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
  },
  benefitValue: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  upgradeHint: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    marginTop: theme.spacing.xl,
    padding: theme.spacing.md,
  },
  upgradeText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: '#92400e',
  },
});

export default PersonalMemberScreen;
