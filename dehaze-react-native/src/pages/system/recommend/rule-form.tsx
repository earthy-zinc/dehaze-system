/**
 * 推荐规则表单
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TextInput, TouchableOpacity, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { RecommendationAPI } from 'dehaze-sdk-js'
import type { RecommendationRule } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemRecommendRuleForm'>;

const SystemRecommendRuleFormScreen: React.FC<Props> = ({ navigation, route }) => {
  const ruleId = route.params?.ruleId;
  const isEdit = !!ruleId;
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<RecommendationRule>({
    ruleName: '', sceneType: '', algorithmIds: [], weight: 50, enabled: true,
  });
  const [algoIdsStr, setAlgoIdsStr] = useState('');

  useEffect(() => {
    if (isEdit && ruleId) {
      setLoading(true);
      RecommendationAPI.getRules().then((rules) => {
        const rule = rules.find((r) => r.id === ruleId);
        if (rule) {
          setForm(rule);
          setAlgoIdsStr(rule.algorithmIds?.join(',') || '');
        }
      }).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
    }
  }, [isEdit, ruleId]);

  const handleSave = async () => {
    if (!form.ruleName.trim()) { Alert.alert('提示', '规则名称必填'); return; }
    setSaving(true);
    try {
      const algorithmIds = algoIdsStr.split(',').map(Number).filter(Boolean);
      await RecommendationAPI.updateRule(ruleId || 0, { ...form, algorithmIds });
      navigation.goBack();
    } catch { Alert.alert('错误', '保存失败'); } finally { setSaving(false); }
  };

  if (loading) return <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑规则' : '新增规则'} showBack onBackPress={() => navigation.goBack()} /><View style={st.center}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={st.flex1}>
      <AppHeader title={isEdit ? '编辑规则' : '新增规则'} showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={st.flex1} contentContainerStyle={st.scrollContent}>
        <F label="规则名称" required><TextInput style={st.input} value={form.ruleName} onChangeText={(v) => setForm((p) => ({ ...p, ruleName: v }))} placeholder="规则名称" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="场景类型"><TextInput style={st.input} value={form.sceneType} onChangeText={(v) => setForm((p) => ({ ...p, sceneType: v }))} placeholder="场景类型" placeholderTextColor={theme.colors.text.tertiary} /></F>
        <F label="算法ID列表（逗号分隔）"><TextInput style={st.input} value={algoIdsStr} onChangeText={setAlgoIdsStr} placeholder="1,2,3" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="权重(0-100)"><TextInput style={st.input} value={String(form.weight)} onChangeText={(v) => setForm((p) => ({ ...p, weight: Number(v) || 0 }))} placeholder="50" placeholderTextColor={theme.colors.text.tertiary} keyboardType="numeric" /></F>
        <F label="状态">
          <View style={st.toggleRow}>
            <TouchableOpacity style={[st.toggleBtn, form.enabled && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, enabled: true }))}><Text style={[st.toggleText, form.enabled && st.toggleTextActive]}>启用</Text></TouchableOpacity>
            <TouchableOpacity style={[st.toggleBtn, !form.enabled && st.toggleBtnActive]} onPress={() => setForm((p) => ({ ...p, enabled: false }))}><Text style={[st.toggleText, !form.enabled && st.toggleTextActive]}>禁用</Text></TouchableOpacity>
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

export default SystemRecommendRuleFormScreen;
