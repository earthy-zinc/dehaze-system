/**
 * 部门表单（新增/编辑）
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { DeptAPI } from 'dehaze-sdk-js'
import type { DeptForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDeptForm'>;

const SystemDeptFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const deptId = route.params?.deptId;
  const isEdit = !!deptId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<DeptForm>({ name: '', parentId: 0, sort: 0, status: 1 });

  useEffect(() => {
    if (isEdit && deptId) {
      setLoading(true);
      DeptAPI.getFormData(deptId).then(setForm).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, deptId]);

  const handleSave = async () => {
    if (!form.name?.trim()) { Alert.alert('提示', '部门名称必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await DeptAPI.update(deptId!, form);
      else await DeptAPI.add(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑部门' : '新增部门'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑部门' : '新增部门'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="部门名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入部门名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="父部门ID"><TextInput style={st.input} value={String(form.parentId ?? 0)} onChangeText={(v) => setForm((p) => ({ ...p, parentId: Number(v) || 0 }))} placeholder="0=根部门" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="排序"><TextInput style={st.input} value={String(form.sort ?? 0)} onChangeText={(v) => setForm((p) => ({ ...p, sort: Number(v) || 0 }))} placeholder="排序号" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="状态">
          <View style={st.toggleRow}>
            <TouchableOpacity style={[st.toggleBtn, form.status === 1 && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 1 }))}><Text style={[st.toggleText, form.status === 1 && st.toggleTextActive]}>启用</Text></TouchableOpacity>
            <TouchableOpacity style={[st.toggleBtn, form.status === 0 && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 0 }))}><Text style={[st.toggleText, form.status === 0 && st.toggleTextActive]}>禁用</Text></TouchableOpacity>
          </View>
        </F>
        <TouchableOpacity style={[st.saveBtn, saving && st.savingOpacity]} onPress={handleSave} disabled={saving}>
          {saving ? <ActivityIndicator size="small" color="#fff" /> : <Text style={st.saveBtnText}>保存</Text>}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const F: React.FC<{ label: string; required?: boolean; children: React.ReactNode }> = ({ label, required, children }) => (
  <View style={st.fieldWrapper}>
    <Text style={st.fieldLabel}>{label}{required && <Text style={st.requiredStar}> *</Text>}</Text>
    {children}
  </View>
);

const st = StyleSheet.create({
  flex1: { flex: 1 },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  fieldWrapper: { marginBottom: theme.spacing.md },
  fieldLabel: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary, marginBottom: theme.spacing.xs },
  requiredStar: { color: theme.colors.status.error },
  input: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light },
  toggleRow: { flexDirection: 'row', gap: theme.spacing.sm },
  toggleBtn: { flex: 1, paddingVertical: theme.spacing.sm, borderRadius: theme.layout.borderRadius.sm, backgroundColor: theme.colors.background.tertiary, alignItems: 'center' },
  toggleBtnActive: { backgroundColor: theme.colors.primary },
  toggleText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary },
  toggleTextActive: { color: '#fff', fontWeight: theme.typography.weights.semibold },
  saveBtn: { marginTop: theme.spacing.lg, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  saveBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
  savingOpacity: { opacity: 0.6 },
});

export default SystemDeptFormScreen;
