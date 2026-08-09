/**
 * 算法管理（管理侧）- 列表/审核上下架
 * 权限：sys:algorithm:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { AlgorithmAPI } from 'dehaze-sdk-js'
import type { Algorithm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemAlgorithm'>;

const STATUS_MAP: Record<number, { label: string; color: string }> = {
  0: { label: '草稿', color: theme.colors.text.tertiary },
  1: { label: '待审核', color: theme.colors.status.warning },
  2: { label: '已发布', color: theme.colors.status.success },
  3: { label: '已驳回', color: theme.colors.status.error },
  4: { label: '已下架', color: theme.colors.text.secondary },
  5: { label: '归档', color: theme.colors.text.secondary },
};

const SystemAlgorithmScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [list, setList] = useState<Algorithm[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [keyword, setKeyword] = useState('');

  const flattenTree = useCallback((nodes: Algorithm[]): Algorithm[] => {
    const result: Algorithm[] = [];
    const walk = (items: Algorithm[]) => {
      items.forEach((item) => {
        result.push(item);
        if (item.children) walk(item.children);
      });
    };
    walk(nodes);
    return result;
  }, []);

  const fetchList = useCallback(async () => {
    try {
      const data = await AlgorithmAPI.getList({ keywords: keyword || undefined });
      setList(flattenTree(data ?? []));
    } catch { Alert.alert('错误', '加载算法列表失败'); }
  }, [keyword, flattenTree]);

  useEffect(() => { setLoading(true); fetchList().finally(() => setLoading(false)); }, [fetchList]);

  const handleAudit = (item: Algorithm) => {
    if (!hasPerm('sys:algorithm:audit')) {
      Alert.alert('提示', '无审核权限');
      return;
    }
    navigation.navigate('SystemAlgorithmAudit', { algorithmId: item.id });
  };

  const handleToggleStatus = (item: Algorithm) => {
    const newStatus = item.status === 2 ? 4 : 2; // 已发布->下架, 否则->上架
    const action = newStatus === 2 ? '上架' : '下架';
    Alert.alert(`确认${action}`, `确定要${action}算法"${item.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await AlgorithmAPI.updateStatus(item.id, newStatus); fetchList(); } catch { Alert.alert('错误', '操作失败'); }
      }},
    ]);
  };

  const handleDelete = (item: Algorithm) => {
    Alert.alert('确认删除', `确定要删除算法"${item.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await AlgorithmAPI.deleteByIds([String(item.id)]); fetchList(); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: Algorithm }) => {
    const st = STATUS_MAP[item.status ?? 0] || { label: '未知', color: theme.colors.text.tertiary };
    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <View style={styles.cardContent}>
            <Text style={styles.cardName}>{item.name}</Text>
            <Text style={styles.cardMeta}>{item.type} · v{item.version || '-'} · {item.size || '-'}</Text>
          </View>
          <View style={[styles.statusBadge, { backgroundColor: st.color + '20' }]}>
            <Text style={[styles.statusText, { color: st.color }]}>{st.label}</Text>
          </View>
        </View>
        <View style={styles.cardActions}>
          {item.status === 1 && hasPerm('sys:algorithm:audit') && (
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleAudit(item)}>
              <Ionicons name="checkmark-done-outline" size={16} color={theme.colors.primary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:algorithm:edit') && (
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleToggleStatus(item)}>
              <Ionicons name={item.status === 2 ? 'arrow-down-circle-outline' : 'arrow-up-circle-outline'} size={16} color={theme.colors.secondary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:algorithm:delete') && (
            <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
              <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <AppHeader title="算法管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索算法" placeholderTextColor={theme.colors.text.tertiary} value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
          {hasPerm('sys:algorithm:add') && (
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemAlgorithmForm', {})}>
              <Ionicons name="add" size={20} color="#fff" />
            </TouchableOpacity>
          )}
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="git-network-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无算法</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  searchBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, gap: theme.spacing.sm },
  searchInputWrap: { flex: 1, flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.sm, height: 40, gap: 6 },
  searchInput: { flex: 1, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, padding: 0 },
  addBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: theme.colors.primary, justifyContent: 'center', alignItems: 'center' },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardContent: { flex: 1 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  cardName: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  cardActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: theme.spacing.sm, marginTop: theme.spacing.sm, paddingTop: theme.spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.colors.border.light },
  actionBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemAlgorithmScreen;
