/**
 * 反馈评价管理（管理侧）- 列表/回复/处理
 * 权限：sys:feedback:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { FeedbackAPI } from 'dehaze-sdk-js'
import type { FeedbackPageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemFeedback'>;

const PAGE_SIZE = 15;
const STATUS_MAP: Record<string, { label: string; color: string }> = {
  pending: { label: '待处理', color: theme.colors.status.warning },
  processing: { label: '处理中', color: theme.colors.primary },
  replied: { label: '已回复', color: theme.colors.status.success },
  closed: { label: '已关闭', color: theme.colors.text.secondary },
};

const SystemFeedbackScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useAuthStore(s => (s.userInfo?.perms ?? []).includes('sys:feedback:*'));

  const [list, setList] = useState<FeedbackPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [keyword, setKeyword] = useState('');

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await FeedbackAPI.listFeedback({ pageNum: pn, pageSize: PAGE_SIZE, keywords: keyword || undefined });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, [keyword]);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const renderItem = ({ item }: { item: FeedbackPageVO }) => {
    const st = STATUS_MAP[item.status] || { label: item.status, color: theme.colors.text.tertiary };
    return (
      <TouchableOpacity style={styles.card} activeOpacity={0.7} onPress={() => navigation.navigate('SystemFeedbackDetail', { feedbackId: item.id })}>
        <View style={styles.cardHeader}>
          <View style={styles.cardBody}>
            <Text style={styles.cardTitle} numberOfLines={1}>{item.title}</Text>
            <Text style={styles.cardMeta}>{item.feedbackType} · {item.username} · {item.createTime}</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: st.color + '20' }]}>
            <Text style={[styles.statusText, { color: st.color }]}>{st.label}</Text>
          </View>
        </View>
        <Text style={styles.cardContent} numberOfLines={2}>{item.content}</Text>
      </TouchableOpacity>
    );
  };

  if (!hasPerm) {
    return (
      <View style={styles.container}>
        <AppHeader title="反馈管理" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.noPerm}>
          <Text style={styles.noPermText}>无权限访问</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="反馈管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索反馈" placeholderTextColor={theme.colors.text.tertiary} value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="chatbox-ellipses-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无反馈</Text></View> : null}
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
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardBody: { flex: 1 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary, flex: 1, marginRight: theme.spacing.sm },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  footerLoader: { padding: theme.spacing.md },
  cardContent: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary, marginTop: theme.spacing.xs, lineHeight: 20 },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemFeedbackScreen;
