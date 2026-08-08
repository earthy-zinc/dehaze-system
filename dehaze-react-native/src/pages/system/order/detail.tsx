/**
 * 订单详情
 */
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { OrderAPI } from 'dehaze-sdk-js'
import type { OrderDetailVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemOrderDetail'>;

const SystemOrderDetailScreen: React.FC<Props> = ({ navigation, route }) => {
  const { orderNo } = route.params;
  const [detail, setDetail] = useState<OrderDetailVO | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    OrderAPI.getDetail(orderNo).then(setDetail).catch(() => Alert.alert('错误', '加载失败')).finally(() => setLoading(false));
  }, [orderNo]);

  if (loading) return <View style={styles.container}>
      <AppHeader title="订单详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><ActivityIndicator size="large" color={theme.colors.primary} /></View></View>;
  if (!detail) return <View style={styles.container}>
      <AppHeader title="订单详情" showBack onBackPress={() => navigation.goBack()} /><View style={styles.centered}><Text>无数据</Text></View></View>;

  return (
    <View style={styles.container}>
      <AppHeader title="订单详情" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.card}>
          <R label="订单号" value={detail.orderNo} />
          <R label="用户" value={detail.username} />
          <R label="套餐" value={detail.packageName} />
          <R label="原价" value={`¥${detail.originalPrice}`} />
          <R label="实付" value={`¥${detail.payableAmount}`} />
          <R label="状态" value={detail.status} />
          <R label="创建时间" value={detail.createTime} />
          {detail.paidTime && <R label="支付时间" value={detail.paidTime} />}
        </View>
      </ScrollView>
    </View>
  );
};

const R: React.FC<{ label: string; value: string }> = ({ label, value }) => (
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

export default SystemOrderDetailScreen;
