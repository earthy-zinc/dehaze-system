/**
 * 算法表单（新增/编辑）
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { AlgorithmAPI } from 'dehaze-sdk-js'
import type { Algorithm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemAlgorithmForm'>;

const SystemAlgorithmFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const algorithmId = route.params?.algorithmId;
  const isEdit = !!algorithmId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<Partial<Algorithm>>({ name: '', type: '', description: '', parentId: 0, status: 0 });

  useEffect(() => {
    if (isEdit && algorithmId) {
      setLoading(true);
      AlgorithmAPI.getAlgorithmInfoById(algorithmId).then(setForm).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, algorithmId]);

  const handleSave = async () => {
    if (!form.name?.trim()) { Alert.alert('提示', '算法名称必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await AlgorithmAPI.update(algorithmId!, form);
      else await AlgorithmAPI.add(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑算法' : '新增算法'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑算法' : '新增算法'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="算法名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入算法名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="算法类型"><TextInput style={st.input} value={form.type} onChangeText={(v) => setForm((p) => ({ ...p, type: v }))} placeholder="算法类型" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="父ID"><TextInput style={st.input} value={String(form.parentId ?? 0)} onChangeText={(v) => setForm((p) => ({ ...p, parentId: Number(v) || 0 }))} placeholder="0=根节点" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="描述"><TextInput style={st.input} value={form.description} onChangeText={(v) => setForm((p) => ({ ...p, description: v }))} placeholder="算法描述" placeholderTextColor={theme.colors.text.tertiary} multiline /></F>
        <F label="版本"><TextInput style={st.input} value={form.version} onChangeText={(v) => setForm((p) => ({ ...p, version: v }))} placeholder="版本号" placeholderTextColor={theme.colors.text.tertiary} /></F>
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
  saveBtn: { marginTop: theme.spacing.lg, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  saveBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
  savingOpacity: { opacity: 0.6 },
});

export default SystemAlgorithmFormScreen;
