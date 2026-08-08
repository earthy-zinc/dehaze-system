/**
 * 推荐管理（管理侧）- 规则列表/编辑
 * 权限：sys:recommendation:*
 */
import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';
import { RecommendationAPI } from 'dehaze-sdk-js'
import type { RecommendationRule } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemRecommend'>;

const SystemRecommendScreen: React.FC<Props> = ({ navigation }) => {
  const hasPerm = useAuthStore(s => (s.userInfo?.perms ?? []).includes('sys:recommendation:*'));

  const [rules, setRules] = useState<RecommendationRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchRules = async () => {
    try {
      const data = await RecommendationAPI.getRules();
      setRules(data ?? []);
    } catch { Alert.alert('错误', '加载失败'); }
  };

  useEffect(() => { setLoading(true); fetchRules().finally(() => setLoading(false)); }, []);

  const renderItem = ({ item }: { item: RecommendationRule }) => (
    <View style={styles.card}>
      <View style={styles.cardBody}>
        <Text style={styles.cardName}>{item.ruleName}</Text>
        <Text style={styles.cardMeta}>场景: {item.sceneType} · 权重: {item.weight} · 算法: {item.algorithmIds?.join(', ')}</Text>
      </View>
      <View style={[styles.statusBadge, item.enabled ? styles.statusEnabled : styles.statusDisabled]}>
        <Text style={[styles.statusText, item.enabled ? styles.statusTextEnabled : styles.statusTextDisabled]}>{item.enabled ? '启用' : '禁用'}</Text>
      </View>
      <TouchableOpacity style={styles.actionBtn} onPress={() => navigation.navigate('SystemRecommendRuleForm', { ruleId: item.id })}>
        <Ionicons name="create-outline" size={16} color={theme.colors.primary} />
      </TouchableOpacity>
    </View>
  );

  if (!hasPerm) {
    return (
      <View style={styles.container}>
        <AppHeader title="推荐管理" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.noPerm}>
          <Text style={styles.noPermText}>无权限访问</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <AppHeader title="推荐管理" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={rules} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchRules(); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
        ListEmptyComponent={!loading ? <View style={styles.empty}><Ionicons name="bulb-outline" size={48} color={theme.colors.text.tertiary} /><Text style={styles.emptyText}>暂无规则</Text></View> : null}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  noPerm: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  noPermText: { color: theme.colors.text.tertiary, fontSize: theme.typography.sizes.bodySmall },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl, paddingTop: theme.spacing.sm },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm, gap: theme.spacing.sm },
  cardBody: { flex: 1 },
  cardName: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  statusBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: theme.layout.borderRadius.full },
  statusEnabled: { backgroundColor: '#34d39920' },
  statusDisabled: { backgroundColor: '#ef444420' },
  statusText: { fontSize: theme.typography.sizes.tiny, fontWeight: theme.typography.weights.semibold },
  statusTextEnabled: { color: '#34d399' },
  statusTextDisabled: { color: '#ef4444' },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center', gap: theme.spacing.sm },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemRecommendScreen;
