/**
 * 数据集表单（新增/编辑）
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { DatasetAPI } from 'dehaze-sdk-js'
import type { DatasetAddForm, DatasetUpdateForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemDatasetForm'>;

const SystemDatasetFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const datasetId = route.params?.datasetId;
  const isEdit = !!datasetId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ name: '', type: '', description: '', parentId: 0, status: '1' });

  useEffect(() => {
    if (isEdit && datasetId) {
      setLoading(true);
      DatasetAPI.getDatasetInfoById(datasetId).then((data) => setForm({
        name: data.name || '',
        type: data.type || '',
        description: data.description || '',
        parentId: data.parentId ?? 0,
        status: String(data.status ?? 1),
      })).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, datasetId]);

  const handleSave = async () => {
    if (!form.name.trim()) { Alert.alert('提示', '数据集名称必填'); return; }
    setSaving(true);
    try {
      if (isEdit) {
        const data: DatasetUpdateForm = { name: form.name, type: form.type, description: form.description, status: form.status };
        await DatasetAPI.update(datasetId!, data);
      } else {
        const data: DatasetAddForm = { name: form.name, type: form.type, description: form.description, parentId: form.parentId, status: form.status };
        await DatasetAPI.add(data);
      }
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑数据集' : '新增数据集'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑数据集' : '新增数据集'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="数据集名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="类型"><TextInput style={st.input} value={form.type} onChangeText={(v) => setForm((p) => ({ ...p, type: v }))} placeholder="类型" placeholderTextColor={theme.colors.text.tertiary} /></F>
        {!isEdit && <F label="父数据集ID"><TextInput style={st.input} value={String(form.parentId)} onChangeText={(v) => setForm((p) => ({ ...p, parentId: Number(v) || 0 }))} placeholder="0=根" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>}
        <F label="描述"><TextInput style={st.input} value={form.description} onChangeText={(v) => setForm((p) => ({ ...p, description: v }))} placeholder="描述" placeholderTextColor={theme.colors.text.tertiary} multiline /></F>
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

export default SystemDatasetFormScreen;
