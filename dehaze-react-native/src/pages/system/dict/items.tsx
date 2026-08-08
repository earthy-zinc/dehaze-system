/**
 * 字典项列表
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { DictAPI } from 'dehaze-sdk-js'
import type { DictPageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDictItem'>;

const PAGE_SIZE = 15;

const SystemDictItemScreen: React.FC<Props> = ({ navigation, route }) => {
  const { typeCode, typeName } = route.params;
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [list, setList] = useState<DictPageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [keyword, setKeyword] = useState('');

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await DictAPI.getDictPage({ pageNum: pn, pageSize: PAGE_SIZE, typeCode, keywords: keyword || undefined });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, [typeCode, keyword]);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleDelete = (item: DictPageVO) => {
    Alert.alert('确认删除', `确定要删除字典项"${item.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await DictAPI.deleteDictByIds(String(item.id)); fetchList(1); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: DictPageVO }) => (
    <View style={styles.card}>
      <View style={styles.cardContent}>
        <Text style={styles.cardName}>{item.name}</Text>
        <Text style={styles.cardMeta}>值: {item.value} · {item.status === 1 ? '启用' : '禁用'}</Text>
      </View>
      <View style={styles.cardActions}>
        {hasPerm('sys:dict:edit') && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemDictItemForm', { dictItemId: item.id, typeCode })}>
            <Ionicons name="create-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
        )}
        {hasPerm('sys:dict:delete') && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
            <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title={`字典项 - ${typeName}`} showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.content}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索字典项" placeholderTextColor={theme.colors.text.tertiary} value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
          {hasPerm('sys:dict:add') && (
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemDictItemForm', { typeCode })}>
              <Ionicons name="add" size={20} color="#fff" />
            </TouchableOpacity>
          )}
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="list-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无字典项</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { flex: 1 },
  searchBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, gap: theme.spacing.sm },
  searchInputWrap: { flex: 1, flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.sm, height: 40, gap: 6 },
  searchInput: { flex: 1, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, padding: 0 },
  addBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.primary, justifyContent: 'center', alignItems: 'center' },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  cardContent: { flex: 1 },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardName: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.medium, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  cardActions: { flexDirection: 'row', gap: 4 },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  footerLoader: { padding: theme.spacing.md },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemDictItemScreen;
