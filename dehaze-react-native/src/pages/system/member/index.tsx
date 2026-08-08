/**
 * 会员管理（管理侧）- 会员列表/等级/成长日志
 * 权限：sys:member:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { MemberAPI } from 'dehaze-sdk-js'
import type { MemberPageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMember'>;

const PAGE_SIZE = 15;
const LEVEL_MAP: Record<string, string> = { level_0: '普通', level_1: 'VIP1', level_2: 'VIP2', level_3: 'VIP3' };

const SystemMemberScreen: React.FC<Props> = ({ navigation }) => {
  const [list, setList] = useState<MemberPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [keyword, setKeyword] = useState('');

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await MemberAPI.getPage({ pageNum: pn, pageSize: PAGE_SIZE, keywords: keyword || undefined });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, [keyword]);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleFreeze = (item: MemberPageVO) => {
    const newStatus = item.status === 1 ? 0 : 1;
    const action = newStatus === 0 ? '冻结' : '解冻';
    Alert.alert(`确认${action}`, `确定要${action}会员"${item.nickname}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await MemberAPI.updateStatus(item.userId, { status: newStatus as any }); fetchList(1); } catch { Alert.alert('错误', '操作失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: MemberPageVO }) => (
    <TouchableOpacity style={styles.card} activeOpacity={0.7} onPress={() => navigation.navigate('SystemMemberDetail', { userId: item.userId })}>
      <View style={styles.cardContent}>
        <Text style={styles.cardName}>{item.nickname}</Text>
        <Text style={styles.cardMeta}>{LEVEL_MAP[item.levelCode] || item.levelName} · 成长值: {item.growthValue} · {item.status === 1 ? '正常' : '冻结'}</Text>
      </View>
      <View style={styles.cardActions}>
        <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemMemberGrowthLog', { userId: item.userId })}>
          <Ionicons name="trending-up-outline" size={16} color={theme.colors.primary} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleFreeze(item)}>
          <Ionicons name={item.status === 1 ? 'snow-outline' : 'sunny-outline'} size={16} color={item.status === 1 ? theme.colors.status.warning : theme.colors.status.success} />
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="会员管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索会员" placeholderTextColor={theme.colors.text.tertiary} value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.userId)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="diamond-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无会员</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  searchBar: { paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm },
  searchInputWrap: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.sm, height: 40, gap: 6 },
  searchInput: { flex: 1, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, padding: 0 },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  cardContent: { flex: 1 },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardName: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  cardActions: { flexDirection: 'row', gap: 4 },
  footerLoader: { padding: theme.spacing.md },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemMemberScreen;
