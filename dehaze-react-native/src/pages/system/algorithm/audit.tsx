/**
 * 算法审核页
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, TextInput, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { AlgorithmAPI } from 'dehaze-sdk-js'
import type { Algorithm } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemAlgorithmAudit'>;

const SystemAlgorithmAuditScreen: React.FC<Props> = ({ navigation, route }) => {
  const { algorithmId } = route.params;
  const [algo, setAlgo] = useState<Algorithm | null>(null);
  const [loading, setLoading] = useState(true);
  const [remark, setRemark] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    AlgorithmAPI.getAlgorithmInfoById(algorithmId)
      .then(setAlgo)
      .catch(() => Alert.alert('错误', '加载算法信息失败'))
      .finally(() => setLoading(false));
  }, [algorithmId]);

  const handleAudit = async (approved: boolean) => {
    setSaving(true);
    try {
      await AlgorithmAPI.auditAlgorithm(algorithmId, { approved, remark: remark || undefined });
      Alert.alert('成功', approved ? '已通过审核' : '已驳回');
      navigation.goBack();
    } catch { Alert.alert('错误', '审核操作失败'); }
    finally { setSaving(false); }
  };

  if (loading) return <View style={styles.container}>
      <AppHeader title="算法审核" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  return (
    <View style={styles.container}>
      <AppHeader title="算法审核" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {algo && (
          <>
            <View style={styles.card}>
              <Text style={styles.name}>{algo.name}</Text>
              <Text style={styles.meta}>类型: {algo.type} · 版本: {algo.version || '-'}</Text>
              {algo.description && <Text style={styles.desc}>{algo.description}</Text>}
            </View>
            <Text style={styles.label}>审核备注</Text>
            <TextInput style={styles.input} value={remark} onChangeText={setRemark} placeholder="请输入审核备注" placeholderTextColor={theme.colors.text.tertiary} multiline numberOfLines={3} textAlignVertical="top" />
            <View style={styles.btnRow}>
              <TouchableOpacity style={[styles.btn, styles.btnReject, saving && styles.disabledBtn]} onPress={() => handleAudit(false)} disabled={saving}>
                {saving ? <ActivityIndicator size="small" color="#fff" /> : <Text style={styles.btnText}>驳回</Text>}
              </TouchableOpacity>
              <TouchableOpacity style={[styles.btn, styles.btnApprove, saving && styles.disabledBtn]} onPress={() => handleAudit(true)} disabled={saving}>
                {saving ? <ActivityIndicator size="small" color="#fff" /> : <Text style={styles.btnText}>通过</Text>}
              </TouchableOpacity>
            </View>
          </>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scrollView: { flex: 1 },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.md, ...theme.layout.shadows.sm },
  name: { fontSize: theme.typography.sizes.large, fontWeight: theme.typography.weights.bold, color: theme.colors.text.primary },
  meta: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.secondary, marginTop: 4 },
  desc: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, marginTop: theme.spacing.sm, lineHeight: 22 },
  label: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary, marginBottom: theme.spacing.xs },
  input: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.sm, paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, borderWidth: StyleSheet.hairlineWidth, borderColor: theme.colors.border.light, minHeight: 80 },
  btnRow: { flexDirection: 'row', gap: theme.spacing.md, marginTop: theme.spacing.lg },
  btn: { flex: 1, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.md, alignItems: 'center' },
  btnReject: { backgroundColor: theme.colors.status.error },
  btnApprove: { backgroundColor: theme.colors.status.success },
  disabledBtn: { opacity: 0.6 },
  btnText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: '#fff' },
});

export default SystemAlgorithmAuditScreen;
