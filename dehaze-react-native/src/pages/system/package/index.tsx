/**
 * 套餐管理（管理侧）- CRUD/上下架
 * 权限：sys:package:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { PackageAPI } from 'dehaze-sdk-js'
import type { PackagePageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemPackage'>;

const PAGE_SIZE = 15;

const SystemPackageScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [list, setList] = useState<PackagePageVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await PackageAPI.getPage({ pageNum: pn, pageSize: PAGE_SIZE });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleToggleStatus = (item: PackagePageVO) => {
    const newStatus = item.status === 1 ? 0 : 1;
    Alert.alert(`确认${newStatus === 0 ? '下架' : '上架'}`, `确定要${newStatus === 0 ? '下架' : '上架'}套餐"${item.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await PackageAPI.updateStatus(item.id, newStatus); fetchList(1); } catch { Alert.alert('错误', '操作失败'); }
      }},
    ]);
  };

  const handleDelete = (item: PackagePageVO) => {
    Alert.alert('确认删除', `确定要删除套餐"${item.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await PackageAPI.deleteByIds(String(item.id)); fetchList(1); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: PackagePageVO }) => (
    <View style={styles.card}>
      <View style={styles.cardBody}>
        <Text style={styles.cardName}>{item.name}</Text>
        <Text style={styles.cardMeta}>{item.levelName} · {item.period} · ¥{item.salePrice} · 销量: {item.salesCount}</Text>
      </View>
      <View style={[styles.statusBadge, item.status === 1 ? styles.statusOnSale : styles.statusOffSale]}>
        <Text style={[styles.statusText, item.status === 1 ? styles.statusTextOnSale : styles.statusTextOffSale]}>{item.status === 1 ? '在售' : '下架'}</Text>
      </View>
      <View style={styles.cardActions}>
        {hasPerm('sys:package:edit') && (
          <>
            <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemPackageForm', { packageId: item.id })}>
              <Ionicons name="create-outline" size={16} color={theme.colors.primary} />
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleToggleStatus(item)}>
              <Ionicons name={item.status === 1 ? 'arrow-down-outline' : 'arrow-up-outline'} size={16} color={theme.colors.secondary} />
            </TouchableOpacity>
          </>
        )}
        {hasPerm('sys:package:delete') && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
            <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="套餐管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        {hasPerm('sys:package:add') && (
          <View style={styles.topBar}>
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemPackageForm', {})}>
              <Ionicons name="add" size={20} color="#fff" /><Text style={styles.addBtnText}>新增套餐</Text>
            </TouchableOpacity>
          </View>
        )}
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="cube-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无套餐</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  topBar: { padding: theme.spacing.md },
  addBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.sm, paddingHorizontal: theme.spacing.md, alignSelf: 'flex-start', gap: 6 },
  addBtnText: { fontSize: theme.typography.sizes.bodySmall, color: '#fff', fontWeight: theme.typography.weights.semibold },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardBody: { flex: 1 },
  cardName: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { position: 'absolute', top: theme.spacing.md, right: theme.spacing.md, paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusOnSale: { backgroundColor: '#34d39920' },
  statusOffSale: { backgroundColor: '#ef444420' },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  statusTextOnSale: { color: '#34d399' },
  statusTextOffSale: { color: '#ef4444' },
  cardActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: theme.spacing.sm, marginTop: theme.spacing.sm, paddingTop: theme.spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.colors.border.light },
  actionBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  footerLoader: { padding: theme.spacing.md },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemPackageScreen;
