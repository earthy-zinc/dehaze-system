/**
 * 我的额度 (L2)
 *
 * ModelAPI.getQuota 剩余/已用/总量 + 进度条
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
import { ModelAPI } from 'dehaze-sdk-js';
import type { PredictionQuota } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PersonalQuotaScreen: React.FC = () => {
  const navigation = useNavigation();
  const [quota, setQuota] = useState<PredictionQuota | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadQuota = useCallback(async () => {
    try {
      setError(null);
      const q = await ModelAPI.getQuota();
      setQuota(q);
    } catch {
      setQuota(null);
      setError('获取额度信息失败，请重试');
      Alert.alert('加载失败', '获取额度信息失败，请重试');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadQuota();
  }, [loadQuota]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadQuota();
    setRefreshing(false);
  }, [loadQuota]);

  const remaining = quota?.remaining ?? 0;
  const total = quota?.total ?? 0;
  const used = quota?.used ?? 0;
  const progress = total > 0 ? Math.min(1, used / total) : 0;

  if (loading) {
    return (
      <View style={styles.container}>
        <AppHeader title="我的额度" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>加载中...</Text>
        </View>
      </View>
    );
  }

  if (error && !quota) {
    return (
      <View style={styles.container}>
        <AppHeader title="我的额度" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.centered}>
          <Ionicons name="alert-circle-outline" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="我的额度" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
      >
      <View style={styles.card}>
        <View style={styles.headerRow}>
          <Ionicons name="wallet-outline" size={24} color={theme.colors.primary} />
          <Text style={styles.cardTitle}>处理额度</Text>
        </View>

        {/* 进度条 */}
        <View style={styles.progressWrap}>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
          </View>
          <Text style={styles.progressText}>{Math.round(progress * 100)}%</Text>
        </View>

        {/* 数据行 */}
        <View style={styles.quotaGrid}>
          <View style={styles.quotaItem}>
            <Text style={styles.quotaValue}>{remaining}</Text>
            <Text style={styles.quotaLabel}>剩余</Text>
          </View>
          <View style={styles.quotaItem}>
            <Text style={styles.quotaValue}>{used}</Text>
            <Text style={styles.quotaLabel}>已用</Text>
          </View>
          <View style={styles.quotaItem}>
            <Text style={styles.quotaValue}>{total}</Text>
            <Text style={styles.quotaLabel}>总量</Text>
          </View>
        </View>

        {quota?.resetDate ? (
          <Text style={styles.resetHint}>
            下次重置：{new Date(quota.resetDate).toLocaleDateString('zh-CN')}
          </Text>
        ) : null}
      </View>
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, flexGrow: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: theme.spacing.xl },
  loadingText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
  errorText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary, marginTop: theme.spacing.sm, textAlign: 'center' },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    ...theme.layout.shadows.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.lg,
  },
  cardTitle: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  progressWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.lg,
  },
  progressTrack: {
    flex: 1,
    height: 10,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: 5,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 5,
    backgroundColor: theme.colors.primary,
  },
  progressText: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.secondary,
    minWidth: 42,
    textAlign: 'right',
  },
  quotaGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: theme.spacing.md,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: theme.colors.border.light,
  },
  quotaItem: {
    alignItems: 'center',
  },
  quotaValue: {
    fontSize: theme.typography.sizes.h3,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  quotaLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 4,
  },
  resetHint: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    marginTop: theme.spacing.sm,
  },
});

export default PersonalQuotaScreen;
