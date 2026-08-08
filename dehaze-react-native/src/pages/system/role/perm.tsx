/**
 * 角色权限分配页
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';

import { RoleAPI, MenuAPI } from 'dehaze-sdk-js';

import type { MenuVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemRolePerm'>;

const SystemRolePermScreen: React.FC<Props> = ({ navigation, route }) => {
  const { roleId } = route.params;
  const [menus, setMenus] = useState<MenuVO[]>([]);
  const [checkedIds, setCheckedIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [menuData, menuIds] = await Promise.all([
          MenuAPI.getList({}),
          RoleAPI.getRoleMenuIds(roleId),
        ]);
        setMenus(menuData ?? []);
        setCheckedIds(new Set(menuIds ?? []));
      } catch { Alert.alert('错误', '加载失败'); }
      finally { setLoading(false); }
    })();
  }, [roleId]);

  const toggleCheck = (id: number) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await RoleAPI.updateRoleMenus(roleId, Array.from(checkedIds));
      Alert.alert('成功', '权限分配成功');
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); }
    finally { setSaving(false); }
  };

  const menuItemPadding = (depth: number) => ({ paddingLeft: theme.spacing.md + depth * 20 });

  const renderMenuItem = (menu: MenuVO, depth: number = 0) => (
    <View key={menu.id}>
      <TouchableOpacity
        style={[styles.menuItem, menuItemPadding(depth)]}
        onPress={() => toggleCheck(menu.id!)}
        activeOpacity={0.7}
      >
        <View style={[styles.checkbox, checkedIds.has(menu.id!) && styles.checkboxChecked]}>
          {checkedIds.has(menu.id!) && <Text style={styles.checkmark}>✓</Text>}
        </View>
        <Text style={styles.menuName}>{menu.name}</Text>
        {menu.perm && <Text style={styles.menuPerm}>{menu.perm}</Text>}
      </TouchableOpacity>
      {menu.children?.map((child) => renderMenuItem(child, depth + 1))}
    </View>
  );

  if (loading) {
    return <View style={styles.container}>
      <AppHeader title="权限分配" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;
  }

  return (
    <View style={styles.container}>
      <AppHeader title="权限分配" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {menus.map((m) => renderMenuItem(m))}
      </ScrollView>
      <View style={styles.bottomBar}>
        <TouchableOpacity style={[styles.saveBtn, saving && styles.disabledBtn]} onPress={handleSave} disabled={saving}>
          {saving ? <ActivityIndicator size="small" color="#fff" /> : <Text style={styles.saveBtnText}>保存权限</Text>}
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scrollView: { flex: 1 },
  scrollContent: { paddingBottom: theme.spacing.xxxl },
  menuItem: { flexDirection: 'row', alignItems: 'center', paddingVertical: theme.spacing.sm, paddingRight: theme.spacing.md, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: theme.colors.border.light },
  checkbox: { width: 20, height: 20, borderRadius: 4, borderWidth: 2, borderColor: theme.colors.border.light, marginRight: theme.spacing.sm, justifyContent: 'center', alignItems: 'center' },
  checkboxChecked: { backgroundColor: theme.colors.primary, borderColor: theme.colors.primary },
  checkmark: { color: '#fff', fontSize: 12 },
  menuName: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, flex: 1 },
  menuPerm: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary },
  bottomBar: { padding: theme.spacing.md, backgroundColor: theme.colors.background.primary, borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: theme.colors.border.light },
  saveBtn: { backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  disabledBtn: { opacity: 0.6 },
  saveBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
});

export default SystemRolePermScreen;
