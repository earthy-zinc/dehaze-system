/**
 * 订单管理（管理侧）- 后台列表/退款审核/统计
 * 权限：sys:order:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { OrderAPI } from 'dehaze-sdk-js'
import type { OrderPageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemOrder'>;

const PAGE_SIZE = 15;
const STATUS_MAP: Record<string, { label: string; color: string }> = {
  pending: { label: '待支付', color: '#fbbf24' },
  paid: { label: '已支付', color: '#3b82f6' },
  completed: { label: '已完成', color: '#34d399' },
  cancelled: { label: '已取消', color: '#6b7280' },
  refunding: { label: '退款中', color: '#f97316' },
  refunded: { label: '已退款', color: '#ef4444' },
};

const SystemOrderScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useAuthStore(s => (s.userInfo?.perms ?? []).includes('sys:order:*'));

  const [list, setList] = useState<OrderPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [keyword, setKeyword] = useState('');

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await OrderAPI.getPage({ pageNum: pn, pageSize: PAGE_SIZE, keywords: keyword || undefined });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, [keyword]);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const renderItem = ({ item }: { item: OrderPageVO }) => {
    const st = STATUS_MAP[item.status] || { label: item.status, color: '#9ca3af' };
    return (
      <TouchableOpacity style={styles.card} activeOpacity={0.7} onPress={() => navigation.navigate('SystemOrderDetail', { orderNo: item.orderNo })}>
        <View style={styles.cardHeader}>
          <View style={styles.cardContent}>
            <Text style={styles.cardId}>#{item.orderNo}</Text>
            <Text style={styles.cardMeta}>{item.packageName} · ¥{item.payableAmount}</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: st.color + '20' }]}>
            <Text style={[styles.statusText, { color: st.color }]}>{st.label}</Text>
          </View>
        </View>
        <Text style={styles.cardUser}>{item.username}</Text>
      </TouchableOpacity>
    );
  };

  if (!hasPerm) {
    return (
      <View style={styles.container}>
        <AppHeader title="订单管理" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.noPerm}>
          <Text style={styles.noPermText}>无权限访问</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="订单管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索订单号/用户名" placeholderTextColor={theme.colors.text.tertiary} value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
        </View>
        <View style={styles.tabs}>
          <TouchableOpacity style={styles.tabBtn} onPress={() => navigation.navigate('SystemOrderRefund')}>
            <Ionicons name="receipt-outline" size={16} color={theme.colors.primary} />
            <Text style={styles.tabText}>退款审核</Text>
          </TouchableOpacity>
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => i.orderNo}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="receipt-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无订单</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  noPerm: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  noPermText: { color: theme.colors.text.tertiary, fontSize: theme.typography.sizes.bodySmall },
  searchBar: { paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm },
  searchInputWrap: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.sm, height: 40, gap: 6 },
  searchInput: { flex: 1, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, padding: 0 },
  tabs: { flexDirection: 'row', paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.sm, gap: theme.spacing.sm },
  tabBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.primaryLight, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, gap: 4 },
  tabText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.primary, fontWeight: theme.typography.weights.medium },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardContent: { flex: 1 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  cardId: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  cardUser: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.secondary, marginTop: theme.spacing.xs },
  footerLoader: { padding: theme.spacing.md },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemOrderScreen;
