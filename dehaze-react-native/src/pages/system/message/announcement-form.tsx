/**
 * 公告表单
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { AnnouncementAPI } from 'dehaze-sdk-js'
import type { AnnouncementForm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMessageAnnouncementForm'>;

const SystemMessageAnnouncementFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const announcementId = route.params?.announcementId;
  const isEdit = !!announcementId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<AnnouncementForm>({
    title: '', content: '', type: 'system', importance: 0, targetScope: 'all',
  });

  useEffect(() => {
    if (isEdit && announcementId) {
      setLoading(true);
      AnnouncementAPI.getDetail(announcementId).then((data) => setForm({
        title: data.title, content: data.content || '', type: data.type, importance: data.importance,
        targetScope: data.targetScope, targetParams: data.targetParams, sendTime: data.sendTime, expireTime: data.expireTime,
      })).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, announcementId]);

  const handleSave = async () => {
    if (!form.title.trim() || !form.content.trim()) { Alert.alert('提示', '标题和内容必填'); return; }
    setSaving(true);
    try {
      if (isEdit) await AnnouncementAPI.update(announcementId!, form);
      else await AnnouncementAPI.create(form);
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑公告' : '新建公告'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑公告' : '新建公告'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="标题" required><TextInput style={st.input} value={form.title} onChangeText={(v) => setForm((p) => ({ ...p, title: v }))} placeholder="公告标题" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="内容" required><TextInput style={st.inputMultiline} value={form.content} onChangeText={(v) => setForm((p) => ({ ...p, content: v }))} placeholder="公告内容" placeholderTextColor={theme.colors.text.tertiary} multiline textAlignVertical="top" /></F>
        <F label="类型"><TextInput style={st.input} value={form.type} onChangeText={(v) => setForm((p) => ({ ...p, type: v }))} placeholder="system" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="重要性(0-2)"><TextInput style={st.input} value={String(form.importance)} onChangeText={(v) => setForm((p) => ({ ...p, importance: Number(v) || 0 }))} placeholder="0" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
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
  inputMultiline: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light, minHeight: 120 },
  saveBtn: { marginTop: theme.spacing.lg, backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  saveBtnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
  savingOpacity: { opacity: 0.6 },
});

export default SystemMessageAnnouncementFormScreen;
