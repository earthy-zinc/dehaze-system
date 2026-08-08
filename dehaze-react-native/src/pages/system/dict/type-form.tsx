/**
 * 字典类型表单
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { DictAPI } from 'dehaze-sdk-js'
import type { DictTypeForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDictTypeForm'>;

const SystemDictTypeFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const dictTypeId = route.params?.dictTypeId;
  const isEdit = !!dictTypeId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<DictTypeForm>({ name: '', code: '', status: 1 });

  useEffect(() => {
    if (isEdit && dictTypeId) {
      setLoading(true);
      DictAPI.getDictTypeForm(dictTypeId).then(setForm).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, dictTypeId]);

  const handleSave = async () => {
    if (!form.name?.trim() || !form.code?.trim()) { Alert.alert('提示', '名称和编码必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await DictAPI.updateDictType(dictTypeId!, form);
      else await DictAPI.addDictType(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑字典类型' : '新增字典类型'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑字典类型' : '新增字典类型'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="类型名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入类型名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="类型编码" required><TextInput style={st.input} value={form.code} onChangeText={(v) => setForm((p) => ({ ...p, code: v }))} placeholder="请输入类型编码" placeholderTextColor={theme.colors.text.tertiary} editable={!isEdit} /></F>
        <F label="状态">
          <View style={st.toggleRow}>
            <TouchableOpacity style={[st.toggleBtn, form.status === 1 && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 1 }))}><Text style={[st.toggleText, form.status === 1 && st.toggleTextActive]}>启用</Text></TouchableOpacity>
            <TouchableOpacity style={[st.toggleBtn, form.status === 0 && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, status: 0 }))}><Text style={[st.toggleText, form.status === 0 && st.toggleTextActive]}>禁用</Text></TouchableOpacity>
          </View>
        </F>
        <F label="备注"><TextInput style={st.input} value={form.remark ?? ''} onChangeText={(v) => setForm((p) => ({ ...p, remark: v }))} placeholder="备注" placeholderTextColor={theme.colors.text.tertiary} /></F>
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

export default SystemDictTypeFormScreen;
