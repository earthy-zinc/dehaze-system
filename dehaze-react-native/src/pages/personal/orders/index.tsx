/**
 * 我的订单 (L2)
 *
 * OrderAPI.listMy 订单列表 + 状态标签
 */
import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  RefreshControl,
  Alert,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { OrderAPI } from 'dehaze-sdk-js';
import type { MyOrderVO } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PAGE_SIZE = 20;

const ORDER_STATUS_MAP: Record<string, { label: string; color: string }> = {
  PENDING_PAYMENT: { label: '待支付', color: theme.colors.status.warning },
  PAID: { label: '已支付', color: theme.colors.primary },
  PROCESSING: { label: '处理中', color: '#8b5cf6' },
  COMPLETED: { label: '已完成', color: theme.colors.status.success },
  CANCELLED: { label: '已取消', color: theme.colors.text.tertiary },
  REFUNDING: { label: '退款中', color: theme.colors.status.warning },
  REFUNDED: { label: '已退款', color: theme.colors.text.tertiary },
};

const PersonalOrdersScreen: React.FC = () => {
  const navigation = useNavigation();
  const [orders, setOrders] = useState<MyOrderVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);

  const loadOrders = useCallback(async (pageNum = 1, isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else if (pageNum === 1) setLoading(true);

      const result = await OrderAPI.listMy({ pageNum, pageSize: PAGE_SIZE });
      const list = result.list || [];
      if (pageNum === 1) {
        setOrders(list);
      } else {
        setOrders(prev => [...prev, ...list]);
      }
      setHasMore(list.length >= PAGE_SIZE);
      setPage(pageNum);
    } catch {
      Alert.alert('加载失败', '获取订单列表失败，请重试');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadOrders(1);
    }, [loadOrders]),
  );

  const handleRefresh = useCallback(() => loadOrders(1, true), [loadOrders]);
  const handleLoadMore = useCallback(() => {
    if (hasMore && !refreshing) loadOrders(page + 1);
  }, [hasMore, refreshing, page, loadOrders]);

  const renderItem = useCallback(({ item }: { item: MyOrderVO }) => {
    const statusInfo = ORDER_STATUS_MAP[item.status] || { label: item.status, color: theme.colors.text.tertiary };

    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.orderNo} numberOfLines={1}>{item.orderNo}</Text>
          <View style={[styles.statusBadge, { backgroundColor: statusInfo.color + '20' }]}>
            <Text style={[styles.statusText, { color: statusInfo.color }]}>{statusInfo.label}</Text>
          </View>
        </View>
        <Text style={styles.packageName}>{item.packageName}</Text>
        <View style={styles.cardFooter}>
          <Text style={styles.amount}>¥{(item.payableAmount ?? 0).toFixed(2)}</Text>
          <Text style={styles.time}>
            {item.createTime ? new Date(item.createTime).toLocaleDateString('zh-CN') : ''}
          </Text>
        </View>
      </View>
    );
  }, []);

  const renderEmpty = () =>
    !loading ? (
      <View style={styles.empty}>
        <Ionicons name="cart-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无订单</Text>
      </View>
    ) : null;

  return (
    <View style={styles.container}>
      <AppHeader title="我的订单" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={orders}
        renderItem={renderItem}
        keyExtractor={item => item.orderNo}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
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
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  orderNo: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    flex: 1,
    marginRight: theme.spacing.sm,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: theme.layout.borderRadius.sm,
  },
  statusText: {
    fontSize: theme.typography.sizes.tiny,
    fontWeight: theme.typography.weights.semibold,
  },
  packageName: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: 8,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  amount: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.status.error,
  },
  time: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  empty: { alignItems: 'center', paddingVertical: theme.spacing.xxxl },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
});

export default PersonalOrdersScreen;
