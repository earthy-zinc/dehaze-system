/**
 * 我的积分余额 (L2)
 *
 * AiBillingAPI.getBalance 积分余额 + 今日/本月已用
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
import { AiBillingAPI } from 'dehaze-sdk-js';
import type { BalanceVO } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PersonalQuotaScreen: React.FC = () => {
  const navigation = useNavigation();
  const [balance, setBalance] = useState<BalanceVO | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadBalance = useCallback(async () => {
    try {
      setError(null);
      const data = await AiBillingAPI.getBalance();
      setBalance(data);
    } catch {
      setBalance(null);
      setError('获取积分余额失败，请重试');
      Alert.alert('加载失败', '获取积分余额失败，请重试');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadBalance();
  }, [loadBalance]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadBalance();
    setRefreshing(false);
  }, [loadBalance]);

  if (loading) {
    return (
      <View style={styles.container}>
        <AppHeader
          title="我的积分余额"
          showBack
          onBackPress={() => navigation.goBack()}
        />
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text style={styles.loadingText}>加载中...</Text>
        </View>
      </View>
    );
  }

  if (error && !balance) {
    return (
      <View style={styles.container}>
        <AppHeader
          title="我的积分余额"
          showBack
          onBackPress={() => navigation.goBack()}
        />
        <View style={styles.centered}>
          <Ionicons
            name="alert-circle-outline"
            size={48}
            color={theme.colors.text.tertiary}
          />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      </View>
    );
  }

  const dailyLimit = balance?.dailyLimit ?? 0;
  const dailyUsed = balance?.dailyUsed ?? 0;
  const monthlyLimit = balance?.monthlyLimit ?? 0;
  const monthlyUsed = balance?.monthlyUsed ?? 0;

  return (
    <View style={styles.container}>
      <AppHeader
        title="我的积分余额"
        showBack
        onBackPress={() => navigation.goBack()}
      />
      <ScrollView
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={[theme.colors.primary]}
            tintColor={theme.colors.primary}
          />
        }
      >
        {/* 积分余额 */}
        <View style={styles.card}>
          <View style={styles.headerRow}>
            <Ionicons
              name="wallet-outline"
              size={24}
              color={theme.colors.primary}
            />
            <Text style={styles.cardTitle}>积分余额</Text>
          </View>
          <Text style={styles.balanceValue}>
            {balance?.creditsBalance ?? 0}
          </Text>
          {balance?.arrearsStatus && (
            <View style={styles.arrearsRow}>
              <Ionicons
                name="warning-outline"
                size={14}
                color={theme.colors.status.error}
              />
              <Text style={styles.arrearsText}>账户已欠费，请及时充值</Text>
            </View>
          )}
        </View>

        {/* 用量 */}
        <View style={styles.card}>
          <Text style={styles.usageTitle}>积分用量</Text>
          <View style={styles.usageRow}>
            <Text style={styles.usageLabel}>今日已用</Text>
            <Text style={styles.usageValue}>
              {dailyUsed} / {dailyLimit}
            </Text>
          </View>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${
                    dailyLimit > 0
                      ? Math.min(100, (dailyUsed / dailyLimit) * 100)
                      : 0
                  }%`,
                },
              ]}
            />
          </View>
          <View style={styles.usageRow}>
            <Text style={styles.usageLabel}>本月已用</Text>
            <Text style={styles.usageValue}>
              {monthlyUsed} / {monthlyLimit}
            </Text>
          </View>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${
                    monthlyLimit > 0
                      ? Math.min(100, (monthlyUsed / monthlyLimit) * 100)
                      : 0
                  }%`,
                },
              ]}
            />
          </View>
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, flexGrow: 1 },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.xl,
  },
  loadingText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.sm,
  },
  errorText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    marginTop: theme.spacing.sm,
    textAlign: 'center',
  },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    marginBottom: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  cardTitle: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  balanceValue: {
    fontSize: 40,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
  arrearsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: theme.spacing.sm,
  },
  arrearsText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.status.error,
  },
  usageTitle: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
  },
  usageRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.xs,
  },
  usageLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  usageValue: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  progressTrack: {
    height: 8,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: theme.spacing.md,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
    backgroundColor: theme.colors.primary,
  },
});

export default PersonalQuotaScreen;
