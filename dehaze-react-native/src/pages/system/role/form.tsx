/**
 * 角色表单（新增/编辑）
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { RoleAPI } from 'dehaze-sdk-js'
import type { RoleForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemRoleForm'>;

const SystemRoleFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const roleId = route.params?.roleId;
  const isEdit = !!roleId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<RoleForm>({ code: '', name: '', sort: 0, status: 1 });

  useEffect(() => {
    if (isEdit && roleId) {
      setLoading(true);
      RoleAPI.getFormData(roleId).then(setForm).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, roleId]);

  const handleSave = async () => {
    if (!form.name?.trim() || !form.code?.trim()) { Alert.alert('提示', '名称和编码必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await RoleAPI.update(roleId!, form);
      else await RoleAPI.add(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={s.flex1}>
      <AppHeader title={isEdit ? '编辑角色' : '新增角色'} showBack onBackPress={() => navigation.goBack()} /><View style={s.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={s.flex1}>
      <AppHeader title={isEdit ? '编辑角色' : '新增角色'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={s.flex1} contentContainerStyle={s.scrollContent}>
        <Field label="角色名称" required><TextInput style={s.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入角色名称" placeholderTextColor={theme.colors.text.tertiary} /></Field>
        <Field label="角色编码" required><TextInput style={s.input} value={form.code} onChangeText={(v) => setForm((p) => ({ ...p, code: v }))} placeholder="请输入角色编码" placeholderTextColor={theme.colors.text.tertiary} editable={!isEdit} /></Field>
        <Field label="排序"><TextInput style={s.input} value={String(form.sort ?? 0)} onChangeText={(v) => setForm((p) => ({ ...p, sort: Number(v) || 0 }))} placeholder="排序号" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></Field>
        <Field label="状态">
          <View style={s.toggleRow}>
            <TouchableOpacity style={[s.toggleBtn, form.status === 1 && s.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 1 }))}><Text style={[s.toggleText, form.status === 1 && s.toggleTextActive]}>启用</Text></TouchableOpacity>
            <TouchableOpacity style={[s.toggleBtn, form.status === 0 && s.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 0 }))}><Text style={[s.toggleText, form.status === 0 && s.toggleTextActive]}>禁用</Text></TouchableOpacity>
          </View>
        </Field>
        <TouchableOpacity style={[s.saveBtn, saving && s.savingOpacity]} onPress={handleSave} disabled={saving} activeOpacity={0.7}>
          {saving ? <ActivityIndicator size="small" color="#fff" /> : <Text style={s.saveBtnText}>保存</Text>}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

const Field: React.FC<{ label: string; required?: boolean; children: React.ReactNode }> = ({ label, required, children }) => (
  <View style={s.fieldWrapper}>
    <Text style={s.fieldLabel}>{label}{required && <Text style={s.requiredStar}> *</Text>}</Text>
    {children}
  </View>
);

const s = StyleSheet.create({
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

export default SystemRoleFormScreen;
