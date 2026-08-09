/**
 * 菜单管理（管理侧）- 菜单树 + 增删改
 * 权限：sys:menu:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert, ActivityIndicator, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { MenuAPI } from 'dehaze-sdk-js'
import type { MenuVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMenu'>;

const SystemMenuScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [menus, setMenus] = useState<MenuVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchMenus = useCallback(async () => {
    try {
      const data = await MenuAPI.getList({});
      setMenus(data ?? []);
    } catch { Alert.alert('错误', '加载菜单失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchMenus().finally(() => setLoading(false)); }, [fetchMenus]);

  const handleRefresh = async () => { setRefreshing(true); await fetchMenus(); setRefreshing(false); };

  const handleDelete = (menu: MenuVO) => {
    if (!hasPerm('sys:menu:delete')) return;
    Alert.alert('确认删除', `确定要删除菜单"${menu.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await MenuAPI.deleteByIds(String(menu.id)); fetchMenus(); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const renderMenu = (menu: MenuVO, depth: number = 0) => (
    <View key={menu.id}>
      <View style={[styles.menuRow, { paddingLeft: theme.spacing.md + depth * 20 }]}>
        <View style={styles.menuInfo}>
          <Text style={styles.menuName}>{menu.icon && <Text>{menu.icon} </Text>}{menu.name}</Text>
          <Text style={styles.menuMeta}>{menu.path || menu.component || '—'} {menu.perm ? `· ${menu.perm}` : ''}</Text>
        </View>
        <View style={styles.menuActions}>
          {hasPerm('sys:menu:edit') && (
            <TouchableOpacity onPress={() => navigation.navigate('SystemMenuForm', { menuId: menu.id })} style={styles.actionBtn}>
              <Ionicons name="create-outline" size={16} color={theme.colors.primary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:menu:add') && (
            <TouchableOpacity onPress={() => navigation.navigate('SystemMenuForm', {})} style={styles.actionBtn}>
              <Ionicons name="add-outline" size={16} color={theme.colors.secondary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:menu:delete') && (
            <TouchableOpacity onPress={() => handleDelete(menu)} style={styles.actionBtn}>
              <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
            </TouchableOpacity>
          )}
        </View>
      </View>
      {menu.children?.map((child) => renderMenu(child, depth + 1))}
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="菜单管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        {hasPerm('sys:menu:add') && (
          <View style={styles.topBar}>
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemMenuForm', {})}>
              <Ionicons name="add" size={20} color="#fff" /><Text style={styles.addBtnText}>新增菜单</Text>
            </TouchableOpacity>
          </View>
        )}
        <ScrollView
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          contentContainerStyle={{ paddingBottom: theme.spacing.xxxl }}
        >
          {loading ? <ActivityIndicator size="large" color={theme.colors.primary} style={{ marginTop: theme.spacing.xxxl }} /> : menus.map((m) => renderMenu(m))}
        </ScrollView>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  topBar: { padding: theme.spacing.md },
  addBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.sm, paddingHorizontal: theme.spacing.md, alignSelf: 'flex-start', gap: 6 },
  addBtnText: { fontSize: theme.typography.sizes.bodySmall, color: '#fff', fontWeight: theme.typography.weights.semibold },
  menuRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: theme.spacing.sm, paddingRight: theme.spacing.md, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: theme.colors.border.light, backgroundColor: theme.colors.background.primary },
  menuInfo: { flex: 1 },
  menuName: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.medium, color: theme.colors.text.primary },
  menuMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  menuActions: { flexDirection: 'row', gap: 4 },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
});

export default SystemMenuScreen;
