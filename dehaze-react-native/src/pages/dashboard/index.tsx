/**
 * 管理向工作台
 *
 * 统计卡片 + 14 个管理功能快捷入口。
 * 入口位于 profile 管理入口组顶部。
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';
import LinearGradient from 'react-native-linear-gradient';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { AlgorithmAPI, DatasetAPI, TaskAPI, OrderAPI } from 'dehaze-sdk-js';

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDashboard'>;

interface StatCard {
  label: string;
  value: string;
  icon: string;
  color: string;
}

interface QuickEntry {
  label: string;
  icon: string;
  route: keyof ProfileStackParamList;
  perm: string;
  color: string;
}

const QUICK_ENTRIES: QuickEntry[] = [
  { label: '用户管理', icon: 'people-outline', route: 'SystemUser', perm: 'sys:user:*', color: '#3b82f6' },
  { label: '角色管理', icon: 'shield-checkmark-outline', route: 'SystemRole', perm: 'sys:role:*', color: '#6366f1' },
  { label: '菜单管理', icon: 'menu-outline', route: 'SystemMenu', perm: 'sys:menu:*', color: '#8b5cf6' },
  { label: '部门管理', icon: 'business-outline', route: 'SystemDept', perm: 'sys:dept:*', color: '#06b6d4' },
  { label: '字典管理', icon: 'book-outline', route: 'SystemDict', perm: 'sys:dict:*', color: '#14b8a6' },
  { label: '算法管理', icon: 'git-network-outline', route: 'SystemAlgorithm', perm: 'sys:algorithm:*', color: '#f59e0b' },
  { label: '数据集管理', icon: 'folder-open-outline', route: 'SystemDataset', perm: 'sys:dataset:*', color: '#ef4444' },
  { label: '任务管理', icon: 'timer-outline', route: 'SystemTask', perm: 'sys:task:*', color: '#ec4899' },
  { label: '会员管理', icon: 'diamond-outline', route: 'SystemMember', perm: 'sys:member:*', color: '#f97316' },
  { label: '套餐管理', icon: 'cube-outline', route: 'SystemPackage', perm: 'sys:package:*', color: '#84cc16' },
  { label: '订单管理', icon: 'receipt-outline', route: 'SystemOrder', perm: 'sys:order:*', color: '#0ea5e9' },
  { label: '反馈评价', icon: 'chatbox-ellipses-outline', route: 'SystemFeedback', perm: 'sys:feedback:*', color: '#a855f7' },
  { label: '消息管理', icon: 'notifications-outline', route: 'SystemMessage', perm: 'sys:notify:*', color: '#22c55e' },
  { label: '推荐管理', icon: 'bulb-outline', route: 'SystemRecommend', perm: 'sys:recommendation:*', color: '#eab308' },
];

const DashboardScreen: React.FC<Props> = ({ navigation }) => {
  const userInfo = useAuthStore(s => s.userInfo);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [stats, setStats] = useState<StatCard[]>([]);
  const [statsError, setStatsError] = useState(false);

  const perms = useAuthStore(s => s.userInfo?.perms ?? []);
  const hasPerm = useCallback((perm: string) => perms.includes(perm), [perms]);

  const loadStats = useCallback(async () => {
    try {
      const [algoRes, dsRes, taskRes, orderRes] = await Promise.allSettled([
        AlgorithmAPI.getList(),
        DatasetAPI.getList({ pageNum: 1, pageSize: 1 }),
        TaskAPI.getPage({ pageNum: 1, pageSize: 1 }),
        OrderAPI.getStats(),
      ]);

      const algoCount = algoRes.status === 'fulfilled' ? algoRes.value?.length ?? 0 : 0;
      const dsTotal = dsRes.status === 'fulfilled' ? dsRes.value?.total ?? 0 : 0;
      const taskTotal = taskRes.status === 'fulfilled' ? taskRes.value?.total ?? 0 : 0;
      const orderTotal = orderRes.status === 'fulfilled' ? orderRes.value?.totalOrders ?? 0 : 0;

      setStats([
        { label: '算法总数', value: String(algoCount), icon: 'git-network-outline', color: '#f59e0b' },
        { label: '数据集', value: String(dsTotal), icon: 'folder-open-outline', color: '#ef4444' },
        { label: '任务总数', value: String(taskTotal), icon: 'timer-outline', color: '#ec4899' },
        { label: '订单总数', value: String(orderTotal), icon: 'receipt-outline', color: '#0ea5e9' },
      ]);
    } catch {
      setStatsError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadStats();
    setRefreshing(false);
  }, [loadStats]);

  const handleEntry = useCallback(
    (entry: QuickEntry) => {
      (navigation as { navigate: (route: string) => void }).navigate(entry.route);
    },
    [navigation],
  );

  // 过滤无权限入口
  const visibleEntries = QUICK_ENTRIES.filter((e) => hasPerm(e.perm));

  return (
    <View style={styles.screenContainer}>
      <AppHeader title="工作台" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={[theme.colors.primary]}
            tintColor={theme.colors.primary}
          />
        }
      >
        {/* 欢迎横幅 */}
        <LinearGradient
          colors={[theme.colors.primary, theme.colors.gradient.primary[1]]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <Text style={styles.heroGreeting}>管理员工作台</Text>
          <Text style={styles.heroSubtitle}>
            {userInfo?.nickname || '管理员'}，欢迎回来
          </Text>
        </LinearGradient>

        {/* 统计卡片 */}
        {loading ? (
          <View style={styles.loadingWrap}>
            <ActivityIndicator size="small" color={theme.colors.primary} />
          </View>
        ) : (
          <>
            <View style={styles.statRow}>
              {stats.map((stat) => (
                <View key={stat.label} style={styles.statCard}>
                  <View style={[styles.statIconWrap, { backgroundColor: stat.color + '20' }]}>
                    <Ionicons name={stat.icon} size={20} color={stat.color} />
                  </View>
                  <Text style={styles.statValue}>{stat.value}</Text>
                  <Text style={styles.statLabel}>{stat.label}</Text>
                </View>
              ))}
            </View>
            {statsError && (
              <Text style={styles.errorText}>统计数据加载失败</Text>
            )}
          </>
        )}

        {/* 快捷入口 */}
        <View style={styles.sectionWrap}>
          <Text style={styles.sectionTitle}>管理功能</Text>
          {visibleEntries.length > 0 ? (
            <View style={styles.entryGrid}>
              {visibleEntries.map((entry) => (
                <TouchableOpacity
                  key={entry.route}
                  style={styles.entryCard}
                  activeOpacity={0.7}
                  onPress={() => handleEntry(entry)}
                >
                  <View style={[styles.entryIconWrap, { backgroundColor: entry.color + '18' }]}>
                    <Ionicons name={entry.icon} size={24} color={entry.color} />
                  </View>
                  <Text style={styles.entryLabel} numberOfLines={1}>
                    {entry.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <Text style={styles.emptyText}>暂无管理功能权限</Text>
          )}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  screenContainer: { flex: 1, backgroundColor: theme.colors.background.secondary },
  scrollView: { flex: 1 },
  scrollContent: { paddingBottom: theme.spacing.xxxl },
  hero: {
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    paddingVertical: theme.spacing.xl,
    paddingHorizontal: theme.spacing.lg,
    borderRadius: theme.layout.borderRadius.xxl,
    ...theme.layout.shadows.lg,
  },
  heroGreeting: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  heroSubtitle: {
    fontSize: theme.typography.sizes.bodySmall,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 4,
  },
  loadingWrap: {
    paddingVertical: theme.spacing.xl,
    alignItems: 'center',
  },
  statRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.lg,
    gap: theme.spacing.sm,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    alignItems: 'center',
    ...theme.layout.shadows.sm,
  },
  statIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  statValue: {
    fontSize: theme.typography.sizes.h4,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  statLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  sectionWrap: {
    marginTop: theme.spacing.lg,
    paddingHorizontal: theme.spacing.md,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  entryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: theme.spacing.sm,
  },
  entryCard: {
    width: '31%',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.sm,
    alignItems: 'center',
    ...theme.layout.shadows.sm,
  },
  entryIconWrap: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.xs,
  },
  entryLabel: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
    textAlign: 'center',
  },
  emptyText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.xl,
  },
  errorText: {
    textAlign: 'center',
    color: theme.colors.status.error,
    padding: 16,
    fontSize: theme.typography.sizes.bodySmall,
  },
});

export default DashboardScreen;
