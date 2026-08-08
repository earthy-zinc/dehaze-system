/**
 * 部门管理（管理侧）- 部门树 + 增删改
 * 权限：sys:dept:*
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity, Alert, ActivityIndicator, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { DeptAPI } from 'dehaze-sdk-js'
import type { DeptVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDept'>;

const SystemDeptScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);

  const [depts, setDepts] = useState<DeptVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchDepts = useCallback(async () => {
    try { const data = await DeptAPI.getList(); setDepts(data ?? []); }
    catch { Alert.alert('错误', '加载部门失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchDepts().finally(() => setLoading(false)); }, [fetchDepts]);

  const handleDelete = (dept: DeptVO) => {
    if (!hasPerm('sys:dept:delete')) return;
    Alert.alert('确认删除', `确定要删除部门"${dept.name}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await DeptAPI.deleteByIds(String(dept.id)); fetchDepts(); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const renderDept = (dept: DeptVO, depth: number = 0) => (
    <View key={dept.id}>
      <View style={[styles.row, { paddingLeft: theme.spacing.md + depth * 24 }]}>
        <View style={styles.cardContent}>
          <Text style={styles.name}>{dept.name}</Text>
          <Text style={styles.meta}>排序: {dept.sort ?? 0} · {dept.status === 1 ? '启用' : '禁用'}</Text>
        </View>
        <View style={styles.actions}>
          {hasPerm('sys:dept:edit') && (
            <TouchableOpacity onPress={() => navigation.navigate('SystemDeptForm', { deptId: dept.id })} style={styles.actionBtn}>
              <Ionicons name="create-outline" size={16} color={theme.colors.primary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:dept:add') && (
            <TouchableOpacity onPress={() => navigation.navigate('SystemDeptForm', {})} style={styles.actionBtn}>
              <Ionicons name="add-outline" size={16} color={theme.colors.secondary} />
            </TouchableOpacity>
          )}
          {hasPerm('sys:dept:delete') && (
            <TouchableOpacity onPress={() => handleDelete(dept)} style={styles.actionBtn}>
              <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
            </TouchableOpacity>
          )}
        </View>
      </View>
      {dept.children?.map((c) => renderDept(c, depth + 1))}
    </View>
  );

  return (
    <View style={styles.screenContainer}>
      <AppHeader title="部门管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        {hasPerm('sys:dept:add') && (
          <View style={styles.topBar}>
            <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemDeptForm', {})}>
              <Ionicons name="add" size={20} color="#fff" /><Text style={styles.addBtnText}>新增部门</Text>
            </TouchableOpacity>
          </View>
        )}
        <ScrollView
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchDepts(); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          contentContainerStyle={styles.scrollContent}
        >
          {loading ? <ActivityIndicator size="large" color={theme.colors.primary} style={styles.loader} /> : depts.map((d) => renderDept(d))}
        </ScrollView>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  screenContainer: { flex: 1, backgroundColor: theme.colors.background.secondary },
  container: { flex: 1 },
  topBar: { padding: theme.spacing.md },
  addBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.sm, paddingHorizontal: theme.spacing.md, alignSelf: 'flex-start', gap: 6 },
  addBtnText: { fontSize: theme.typography.sizes.bodySmall, color: '#fff', fontWeight: theme.typography.weights.semibold },
  cardContent: { flex: 1 },
  scrollContent: { paddingBottom: theme.spacing.xxxl },
  loader: { marginTop: theme.spacing.xxxl },
  row: { flexDirection: 'row', alignItems: 'center', paddingVertical: theme.spacing.sm, paddingRight: theme.spacing.md, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: theme.colors.border.light, backgroundColor: theme.colors.background.primary },
  name: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.medium, color: theme.colors.text.primary },
  meta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  actions: { flexDirection: 'row', gap: 4 },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
});

export default SystemDeptScreen;
