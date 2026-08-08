/**
 * 会员详情
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, Alert, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { MemberAPI } from 'dehaze-sdk-js'
import type { MemberDetailVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMemberDetail'>;

const SystemMemberDetailScreen: React.FC<Props> = ({ navigation, route }) => {
  const { userId } = route.params;
  const [detail, setDetail] = useState<MemberDetailVO | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    MemberAPI.getDetail(userId).then(setDetail).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
  }, [userId]);

  if (loading) return <View style={styles.container}>
      <AppHeader title="会员详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;

  if (!detail) return <View style={styles.container}>
      <AppHeader title="会员详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><Text>无数据</Text></View></View>;

  return (
    <View style={styles.container}>
      <AppHeader title="会员详情" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.card}>
          <Row label="用户ID" value={String(detail.userId)} />
          <Row label="昵称" value={detail.nickname} />
          <Row label="等级" value={detail.levelName} />
          <Row label="成长值" value={String(detail.growthValue)} />
          <Row label="进度" value={`${detail.progressPercent}%`} />
          <Row label="状态" value={detail.status === 1 ? '正常' : '冻结'} />
          <Row label="过期时间" value={detail.expireTime || '-'} />
          <Row label="月去雾额度" value={`${detail.monthlyDehazeUsed}/${detail.monthlyDehazeQuota}`} />
        </View>
      </ScrollView>
    </View>
  );
};

const Row: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <View style={styles.row}>
    <Text style={styles.label}>{label}</Text>
    <Text style={styles.value}>{value}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: { flex: 1 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  scrollView: { flex: 1 },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, ...theme.layout.shadows.sm },
  row: { flexDirection: 'row', paddingVertical: 10, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: theme.colors.border.light },
  label: { width: 80, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
  value: { flex: 1, fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.primary, fontWeight: theme.typography.weights.medium },
});

export default SystemMemberDetailScreen;
