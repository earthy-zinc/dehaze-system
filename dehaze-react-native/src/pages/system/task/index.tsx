/**
 * 任务管理（管理侧）- 全用户任务列表/取消/重试
 * 权限：sys:task:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { TaskAPI } from 'dehaze-sdk-js'
import type { TaskVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemTask'>;

const PAGE_SIZE = 15;
const STATUS_MAP: Record<number, { label: string; color: string }> = {
  1: { label: '待处理', color: theme.colors.status.warning },
  2: { label: '处理中', color: theme.colors.primary },
  3: { label: '已完成', color: theme.colors.status.success },
  4: { label: '失败', color: theme.colors.status.error },
  5: { label: '已取消', color: theme.colors.text.secondary },
};

const SystemTaskScreen: React.FC<Props> = ({ navigation }) => {
  const [list, setList] = useState<TaskVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await TaskAPI.getPage({ pageNum: pn, pageSize: PAGE_SIZE });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleCancel = (task: TaskVO) => {
    Alert.alert('确认取消', `确定要取消任务"${task.taskId}"吗？`, [
      { text: '否', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await TaskAPI.cancel(task.taskId); fetchList(1); } catch { Alert.alert('错误', '取消失败'); }
      }},
    ]);
  };

  const handleRetry = (task: TaskVO) => {
    Alert.alert('确认重试', `确定要重试任务"${task.taskId}"吗？`, [
      { text: '否', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await TaskAPI.retry(task.taskId); fetchList(1); } catch { Alert.alert('错误', '重试失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: TaskVO }) => {
    const st = STATUS_MAP[item.status] || { label: '未知', color: theme.colors.text.tertiary };
    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <View style={styles.cardContent}>
            <Text style={styles.cardId}>#{item.taskId}</Text>
            <Text style={styles.cardMeta}>{item.taskType || '-'} · 进度: {item.progress}%</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: st.color + '20' }]}>
            <Text style={[styles.statusText, { color: st.color }]}>{st.label}</Text>
          </View>
        </View>
        {item.error && <Text style={styles.errorText}>错误: {item.error}</Text>}
        <View style={styles.cardActions}>
          {(item.status === 1 || item.status === 2) && (
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleCancel(item)}>
              <Ionicons name="close-circle-outline" size={16} color={theme.colors.status.warning} />
            </TouchableOpacity>
          )}
          {item.status === 4 && (
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleRetry(item)}>
              <Ionicons name="refresh-outline" size={16} color={theme.colors.primary} />
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <AppHeader title="任务管理" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={list} renderItem={renderItem} keyExtractor={(i) => i.taskId}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
        onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
        onEndReachedThreshold={0.3}
        ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
        ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="timer-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无任务</Text></View> : null}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { flex: 1 },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl, paddingTop: theme.spacing.sm },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardContent: { flex: 1 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  cardId: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  footerLoader: { padding: theme.spacing.md },
  errorText: { fontSize: theme.typography.sizes.tiny, color: theme.colors.status.error, marginTop: theme.spacing.xs },
  cardActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: theme.spacing.sm, marginTop: theme.spacing.sm },
  actionBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemTaskScreen;
