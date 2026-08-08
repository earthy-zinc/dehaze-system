/**
 * 消息模板表单
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { MessageTemplateAPI } from 'dehaze-sdk-js'
import type { MessageTemplateForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMessageTemplateForm'>;

const SystemMessageTemplateFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const templateId = route.params?.templateId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<MessageTemplateForm>({ name: '', titleTemplate: '', status: 1 });

  useEffect(() => {
    if (templateId) {
      setLoading(true);
      MessageTemplateAPI.getDetail(templateId).then((data) => setForm({
        name: data.name, titleTemplate: data.titleTemplate, contentTemplate: data.contentTemplate,
        priority: data.priority, status: data.status,
      })).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [templateId]);

  const handleSave = async () => {
    if (!form.name?.trim()) { Alert.alert('提示', '模板名称必填'); return; }
    setSaving(true);
    try {
      await MessageTemplateAPI.update(templateId!, form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title="编辑模板" showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title="编辑模板" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="模板名称" required><TextInput style={st.input} value={form.name} onChangeText={(v) => setForm((p) => ({ ...p, name: v }))} placeholder="模板名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="标题模板"><TextInput style={st.input} value={form.titleTemplate} onChangeText={(v) => setForm((p) => ({ ...p, titleTemplate: v }))} placeholder="标题模板" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="内容模板"><TextInput style={st.inputMultiline} value={form.contentTemplate} onChangeText={(v) => setForm((p) => ({ ...p, contentTemplate: v }))} placeholder="内容模板" placeholderTextColor={theme.colors.text.tertiary} multiline textAlignVertical="top" /></F>
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
  inputMultiline: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light, minHeight: 100 },
  saveBtn: { marginTop: theme.spacing.lg, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  saveBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
  savingOpacity: { opacity: 0.6 },
});

export default SystemMessageTemplateFormScreen;
