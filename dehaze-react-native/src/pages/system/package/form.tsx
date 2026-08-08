/**
 * 套餐表单
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { PackageAPI } from 'dehaze-sdk-js'
import type { PackageForm, PackageLevelCode, PackagePeriod } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemPackageForm'>;

const SystemPackageFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const packageId = route.params?.packageId;
  const isEdit = !!packageId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<PackageForm>({
    name: '', levelCode: 'level_1' as PackageLevelCode, period: 'monthly' as PackagePeriod,
    periodDays: 30, originalPrice: 0, salePrice: 0, status: 1,
  });

  useEffect(() => {
    if (isEdit && packageId) {
      setLoading(true);
      PackageAPI.getForm(packageId).then(setForm).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, packageId]);

  const handleSave = async () => {
    if (!form.name.trim()) { Alert.alert('提示', '套餐名称必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await PackageAPI.update(packageId!, form);
      else await PackageAPI.add(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑套餐' : '新增套餐'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑套餐' : '新增套餐'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="套餐名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="请输入名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="原价"><TextInput style={st.input} value={String(form.originalPrice)} onChangeText={(v) => setForm((p) => ({ ...p, originalPrice: Number(v) || 0 }))} placeholder="原价" placeholderTextColor={theme.colors.text.tertiary} keyboardType="decimal-pad" /></F>
        <F label="售价"><TextInput style={st.input} value={String(form.salePrice)} onChangeText={(v) => setForm((p) => ({ ...p, salePrice: Number(v) || 0 }))} placeholder="售价" placeholderTextColor={theme.colors.text.tertiary} keyboardType="decimal-pad" /></F>
        <F label="周期天数"><TextInput style={st.input} value={String(form.periodDays)} onChangeText={(v) => setForm((p) => ({ ...p, periodDays: Number(v) || 30 }))} placeholder="天数" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="描述"><TextInput style={st.input} value={form.description ?? ''} onChangeText={(v) => setForm((p) => ({ ...p, description: v }))} placeholder="描述" placeholderTextColor={theme.colors.text.tertiary} multiline /></F>
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

export default SystemPackageFormScreen;
