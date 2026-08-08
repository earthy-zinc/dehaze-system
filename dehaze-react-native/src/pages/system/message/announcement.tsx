/**
 * 公告管理
 */
import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert, RefreshControl } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import { AnnouncementAPI } from 'dehaze-sdk-js'
import type { AnnouncementVO } from 'dehaze-sdk-js'

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMessageAnnouncement'>;

const PAGE_SIZE = 15;

const SystemMessageAnnouncementScreen: React.FC<Props> = ({ navigation }) => {
  const [list, setList] = useState<AnnouncementVO[]>([]);
  const [total, setTotal] = useState(0);
  const [pageNum, setPageNum] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const fetchList = useCallback(async (pn: number) => {
    try {
      const res = await AnnouncementAPI.getPage({ pageNum: pn, pageSize: PAGE_SIZE });
      const fetched = res?.list ?? [];
      if (pn === 1) setList(fetched); else setList((prev) => [...prev, ...fetched]);
      setTotal(res?.total ?? 0);
      setPageNum(pn);
    } catch { Alert.alert('错误', '加载失败'); }
  }, []);

  useEffect(() => { setLoading(true); fetchList(1).finally(() => setLoading(false)); }, [fetchList]);

  const handleDelete = (item: AnnouncementVO) => {
    Alert.alert('确认删除', `确定要删除公告"${item.title}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: async () => {
        try { await AnnouncementAPI.deleteById(item.id); fetchList(1); } catch { Alert.alert('错误', '删除失败'); }
      }},
    ]);
  };

  const handleSend = (item: AnnouncementVO) => {
    Alert.alert('确认发送', `确定要发送公告"${item.title}"吗？`, [
      { text: '取消', style: 'cancel' },
      { text: '确定', onPress: async () => {
        try { await AnnouncementAPI.send(item.id); fetchList(1); Alert.alert('成功', '公告已发送'); } catch { Alert.alert('错误', '发送失败'); }
      }},
    ]);
  };

  const renderItem = ({ item }: { item: AnnouncementVO }) => (
    <View style={styles.card}>
      <View style={styles.cardContent}>
        <Text style={styles.cardTitle}>{item.title}</Text>
        <Text style={styles.cardMeta}>{item.typeLabel || item.type} · 状态: {item.statusLabel || item.status} · 已发: {item.sentCount ?? 0}</Text>
      </View>
      <View style={styles.cardActions}>
        {item.status !== 2 && (
          <TouchableOpacity style={styles.actionBtn} onPress={() => handleSend(item)}>
            <Ionicons name="send-outline" size={16} color={theme.colors.primary} />
          </TouchableOpacity>
        )}
        <TouchableOpacity style={styles.actionBtn} onPress={() => handleDelete(item)}>
          <Ionicons name="trash-outline" size={16} color={theme.colors.status.error} />
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <AppHeader title="公告管理" showBack onBackPress={() => navigation.goBack()} />
      <View style={styles.container}>
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.addBtn} onPress={() => navigation.navigate('SystemMessageAnnouncementForm', {})}>
            <Ionicons name="add" size={20} color="#fff" /><Text style={styles.addBtnText}>新建公告</Text>
          </TouchableOpacity>
        </View>
        <FlatList
          data={list} renderItem={renderItem} keyExtractor={(i) => String(i.id)}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={async () => { setRefreshing(true); await fetchList(1); setRefreshing(false); }} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />}
          onEndReached={async () => { if (!loadingMore && list.length < total) { setLoadingMore(true); await fetchList(pageNum + 1); setLoadingMore(false); } }}
          onEndReachedThreshold={0.3}
          ListFooterComponent={loadingMore ? <ActivityIndicator size="small" color={theme.colors.primary} style={styles.footerLoader} /> : null}
          ListEmptyComponent={!loading ? <View style={styles.empty}><Text style={styles.emptyText}>暂无公告</Text></View> : null}
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  topBar: { padding: theme.spacing.md },
  addBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.primary, borderRadius: theme.layout.borderRadius.md, paddingVertical: theme.spacing.sm, paddingHorizontal: theme.spacing.md, alignSelf: 'flex-start', gap: 6 },
  addBtnText: { fontSize: theme.typography.sizes.bodySmall, color: '#fff', fontWeight: theme.typography.weights.semibold },
  listContent: { paddingHorizontal: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  cardContent: { flex: 1 },
  card: { flexDirection: 'row', alignItems: 'center', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.md, marginBottom: theme.spacing.sm, ...theme.layout.shadows.sm },
  cardTitle: { fontSize: theme.typography.sizes.bodySmall, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  cardMeta: { fontSize: theme.typography.sizes.tiny, color: theme.colors.text.tertiary, marginTop: 2 },
  cardActions: { flexDirection: 'row', gap: 4 },
  actionBtn: { width: 30, height: 30, borderRadius: 15, backgroundColor: theme.colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  footerLoader: { padding: theme.spacing.md },
  empty: { paddingVertical: theme.spacing.xxxl, alignItems: 'center' },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
});

export default SystemMessageAnnouncementScreen;
