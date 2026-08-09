/**
 * 角色管理（管理侧）
 * 权限：sys:role:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput,
  Alert, RefreshControl,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { RoleAPI } from 'dehaze-sdk-js'
import type { RolePageVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemRole'>;

const SystemRoleScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [list, setList] = useState<RolePageVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [keyword, setKeyword] = useState('');

  const fetchList = useCallback(async () => {
    try {
      const res = await RoleAPI.getPage({ keywords: keyword || undefined });
      setList(res?.list ?? []);
    } catch {
      Alert.alert('错误', '加载角色列表失败');
    }
  }, [keyword]);

  useEffect(() => {
    setLoading(true);
    fetchList().finally(() => setLoading(false));
  }, [fetchList]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchList();
    setRefreshing(false);
  };

  const handleDelete = (role: RolePageVO) => {
    if (!hasPerm('sys:role:delete')) return;
    Alert.alert('确认删除', `确定要删除角色"${role.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      {
        text: '确定', style: 'destructive',
        onPress: async () => {
          try { await RoleAPI.deleteByIds(String(role.id)); fetchList(); }
          catch { Alert.alert('错误', '删除失败'); }
        },
      },
    ]);
  };

  const renderItem = ({ item }: { item: RolePageVO }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardInfo}>
          <Text style={styles.cardName}>{item.name}</Text>
          <Text style={styles.cardCode}>{item.code}</Text>
        </View>
        <View style={[styles.statusBadge, item.status === 1 ? styles.statusEnabled : styles.statusDisabled]}>
          <Text style={[styles.statusText, item.status === 1 ? styles.statusTextEnabled : styles.statusTextDisabled]}>
            {item.status === 1 ? '启用' : '禁用'}
          </Text>
        </View>
      </View>
      <View style={styles.cardActions}>
        {hasPerm('sys:role:edit') && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemRoleForm', { roleId: item.id })}>
            <Ionicons name="create-outline" size={18} color={theme.colors.primary} />
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemRolePerm', { roleId: item.id! })}>
          <Ionicons name="shield-outline" size={18} color={theme.colors.gradient.primary[1]} />
        </TouchableOpacity>
        {hasPerm('sys:role:delete') && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
            <Ionicons name="trash-outline" size={18} color={theme.colors.status.error} />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="角色管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.searchBar}>
          <View style={styles.searchInputWrap}>
            <Ionicons name="search-outline" size={18} color={theme.colors.text.tertiary} />
            <TextInput style={styles.searchInput} placeholder="搜索角色" placeholderTextColor={theme.colors.text.tertiary}
              value={keyword} onChangeText={setKeyword} returnKeyType="search" />
          </View>
          {hasPerm('sys:role:add') && (
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemRoleForm', {})}>
              <Ionicons name="add" size={20} color="#fff" />
            </TouchableOpacity>
          )}
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="shield-checkmark-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无角色</Text></View> : null}
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
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  cardInfo: { flex: 1 },
  cardName: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardCode: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: theme.layout.borderRadius.full },
  statusEnabled: { backgroundColor: '#34d39920' },
  statusDisabled: { backgroundColor: '#ef444420' },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  statusTextEnabled: { color: '#34d399' },
  statusTextDisabled: { color: '#ef4444' },
  cardActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: theme.spacing.sm, marginTop: theme.spacing.sm, paddingTop: theme.spacing.sm, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.colors.border.light },
  actionBtn: { width: 36, height: 36, borderRadius: 18, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemRoleScreen;
