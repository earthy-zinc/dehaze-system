/**
 * 退款审核列表
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { OrderAPI } from 'dehaze-sdk-js'
import type { RefundRecordVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemOrderRefund'>;

const PAGE_SIZE = 15;

const SystemOrderRefundScreen: React.FC<Props> = ({ navigation }) => {
  const [list, setList] = useState<RefundRecordVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await OrderAPI.listRefunds({ pageNum: pn, pageSize: PAGE_SIZE });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleApprove = (item: RefundRecordVO) => {
    Alert.alert('确认通过', `确定要通过退款"${item.refundNo}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await OrderAPI.approveRefund(item.id, { approved: true, remark: '' }); fetchList(1); } catch { Alert.alert('错误', '操作失败'); }
      }},
    ]);
  };

  const handleReject = (item: RefundRecordVO) => {
    Alert.prompt ? Alert.prompt('驳回原因', '请输入驳回原因', async (text) => {
      try { await OrderAPI.rejectRefund(item.id, { approved: false, remark: text || '驳回' }); fetchList(1); } catch { Alert.alert('错误', '操作失败'); }
    }) : (
      Alert.alert('确认驳回', `确定要驳回退款"${item.refundNo}"吗？`, [
        { text: '取消', style: 'cancel' },
        { text: '确定', onPress: async () => {
          try { await OrderAPI.rejectRefund(item.id, { approved: false, remark: '驳回' }); fetchList(1); } catch { Alert.alert('错误', '操作失败'); }
        }},
      ])
    );
  };

  const renderItem = ({ item }: { item: RefundRecordVO }) => (
    <View style={styles.card}>
      <View style={styles.cardContent}>
        <Text style={styles.cardId}>{item.refundNo}</Text>
        <Text style={styles.cardMeta}>订单: {item.orderNo} · 用户: {item.username} · ¥{item.refundAmount}</Text>
        <Text style={styles.cardMeta}>状态: {item.status} · {item.applyTime}</Text>
      </View>
      {item.status === 'refunding' && (
        <View style={styles.cardActions}>
          <TouchableOpacity style={[styles.btn, styles.btnApprove]} onPress={() => handleApprove(item)}>
            <Text style={styles.btnText}>通过</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.btn, styles.btnReject]} onPress={() => handleReject(item)}>
            <Text style={styles.btnText}>驳回</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="退款审核" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
        onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
        onEndReachedThreshold={0.3}
        ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
        ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="receipt-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无退款申请</Text></View> : null}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { flex: 1 },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl, paddingTop: theme.spacing.sm },
  cardContent: { flex: 1 },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardId: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  cardActions: { flexDirection: 'row', gap: theme.spacing.sm, marginTop: theme.spacing.sm },
  btn: { flex: 1, borderRadius: theme.layout.borderRadius.sm, paddingVertical: theme.spacing.sm, alignItems: 'center' },
  btnApprove: { backgroundColor: theme.colors.status.success },
  btnReject: { backgroundColor: theme.colors.status.error },
  footerLoader: { padding: theme.spacing.md },
  btnText: { fontSize: theme.typography.sizes.bodySmall, color: '#fff', fontWeight: theme.typography.weights.semibold },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemOrderRefundScreen;
