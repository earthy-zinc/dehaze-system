/**
 * 我的套餐 (L2)
 *
 * PackageAPI.listOnSale 套餐卡片 + 购买入口
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { PackageAPI, OrderAPI } from 'dehaze-sdk-js';
import type { PackageDetailVO } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PersonalPackageScreen: React.FC = () => {
  const navigation = useNavigation();
  const [packages, setPackages] = useState<PackageDetailVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadPackages = useCallback(async () => {
    try {
      const list = await PackageAPI.listOnSale();
      setPackages(list || []);
    } catch {
      setPackages([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadPackages();
  }, [loadPackages]);

  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    loadPackages();
  }, [loadPackages]);

  const handleBuy = useCallback((pkg: PackageDetailVO) => {
    Alert.alert('购买套餐', `确认购买「${pkg.name}」？`, [
      { text: '取消', style: 'cancel' },
      {
        text: '确认',
        onPress: async () => {
          try {
            const result = await OrderAPI.create({ packageId: pkg.id, payMethod: 'balance' });
            Alert.alert('下单成功', `订单号：${result.orderNo}\n请前往「我的订单」查看并完成支付`, [
              { text: '确定' },
            ]);
          } catch {
            Alert.alert('下单失败', '创建订单失败，请稍后重试');
          }
        },
      },
    ]);
  }, []);

  const renderItem = useCallback(
    ({ item }: { item: PackageDetailVO }) => {
      const isRecommended = item.salesCount > 100;

      return (
        <View style={[styles.card, isRecommended && styles.recommendedCard]}>
          {isRecommended && (
            <View style={styles.recBadge}>
              <Text style={styles.recBadgeText}>热门</Text>
            </View>
          )}
          <Text style={styles.pkgName}>{item.name}</Text>
          <Text style={styles.pkgLevel}>{item.levelName}</Text>
          {item.description ? (
            <Text style={styles.pkgDesc} numberOfLines={2}>{item.description}</Text>
          ) : null}
          <View style={styles.priceRow}>
            <Text style={styles.price}>¥{item.salePrice?.toFixed(2) ?? '0.00'}</Text>
            {item.originalPrice > item.salePrice ? (
              <Text style={styles.originalPrice}>¥{item.originalPrice.toFixed(2)}</Text>
            ) : null}
            {item.periodDays ? (
              <Text style={styles.duration}>/ {item.periodDays}天</Text>
            ) : null}
          </View>
          <TouchableOpacity
            style={styles.buyBtn}
            onPress={() => handleBuy(item)}
            activeOpacity={0.8}
          >
            <Text style={styles.buyBtnText}>立即购买</Text>
          </TouchableOpacity>
        </View>
      );
    },
    [handleBuy],
  );

  const renderEmpty = () =>
    !loading ? (
      <View style={styles.empty}>
        <Ionicons name="cube-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无可购套餐</Text>
      </View>
    ) : null;

  return (
    <View style={styles.container}>
      <AppHeader title="套餐购买" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={packages}
        renderItem={renderItem}
        keyExtractor={item => String(item.id)}
        numColumns={1}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
        ListEmptyComponent={renderEmpty}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  list: { padding: theme.spacing.md, flexGrow: 1 },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    marginBottom: theme.spacing.md,
    ...theme.layout.shadows.sm,
    position: 'relative',
  },
  recommendedCard: {
    borderWidth: 2,
    borderColor: theme.colors.primary,
  },
  recBadge: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderBottomLeftRadius: theme.layout.borderRadius.sm,
    borderTopRightRadius: theme.layout.borderRadius.lg,
  },
  recBadgeText: {
    fontSize: theme.typography.sizes.tiny,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  pkgName: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: 4,
  },
  pkgLevel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.semibold,
    marginBottom: 8,
  },
  pkgDesc: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.md,
  },
  priceRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 6,
    marginBottom: theme.spacing.md,
  },
  price: {
    fontSize: theme.typography.sizes.h3,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.status.error,
  },
  originalPrice: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    textDecorationLine: 'line-through',
  },
  duration: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  buyBtn: {
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.md,
    paddingVertical: 12,
    alignItems: 'center',
  },
  buyBtnText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  empty: { alignItems: 'center', paddingVertical: theme.spacing.xxxl },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
});

export default PersonalPackageScreen;
