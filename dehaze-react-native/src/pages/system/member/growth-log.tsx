/**
 * 会员成长日志
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { MemberAPI } from 'dehaze-sdk-js'
import type { GrowthLogVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMemberGrowthLog'>;

const PAGE_SIZE = 15;
const TYPE_MAP: Record<string, string> = { dehaze: '去雾', evaluate: '评估', rating: '评分', sign_in: '签到', sign_in_bonus: '签到奖励', consume: '消费', refund_deduct: '退款扣除', admin_adjust: '管理员调整' };

const SystemMemberGrowthLogScreen: React.FC<Props> = ({ navigation }) => {
  const [list, setList] = useState<GrowthLogVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await MemberAPI.getGrowthLogs({ pageNum: pn, pageSize: PAGE_SIZE });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const renderItem = ({ item }: { item: GrowthLogVO }) => (
    <View style={styles.card}>
      <View style={styles.cardContent}>
        <Text style={styles.cardType}>{TYPE_MAP[item.changeType] || item.changeType}</Text>
        <Text style={styles.cardMeta}>{item.reason || '-'} · {item.createTime}</Text>
      </View>
      <Text style={[styles.value, { color: item.changeValue > 0 ? theme.colors.status.success : theme.colors.status.error }]}>
        {item.changeValue > 0 ? '+' : ''}{item.changeValue}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="成长日志" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
        onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
        onEndReachedThreshold={0.3}
        ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
        ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="trending-up-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无日志</Text></View> : null}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { flex: 1 },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl, paddingTop: theme.spacing.sm },
  cardContent: { flex: 1 },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardType: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.medium, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  footerLoader: { padding: theme.spacing.md },
  value: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemMemberGrowthLogScreen;
