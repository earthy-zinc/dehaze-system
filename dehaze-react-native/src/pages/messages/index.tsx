/**
 * 消息 Tab (L1)
 *
 * 对接 MessageAPI：分类筛选、未读角标、全部已读、跳转详情、设置入口。
 * 按 05-菜单与页面层级规划 2.4 节设计。
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NavigationProp } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { MessageAPI } from 'dehaze-sdk-js';
import type { MessageVO } from 'dehaze-sdk-js';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import { useMessagesStore } from '@/store/messages';
import type { RootStackParamList } from '@/routes/types';

type MessageType = 'all' | 'system' | 'processing' | 'activity';

const TYPE_TABS: { key: MessageType; label: string }[] = [
  { key: 'all', label: '全部' },
  { key: 'system', label: '系统' },
  { key: 'processing', label: '处理' },
  { key: 'activity', label: '活动' },
];

const PAGE_SIZE = 20;

export default function MessagesScreen() {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();
  const { unreadCount, setUnreadCount, decrementUnread } = useMessagesStore();

  const [activeType, setActiveType] = useState<MessageType>('all');
  const [messages, setMessages] = useState<MessageVO[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [pageNum, setPageNum] = useState(1);

  useEffect(() => {
    navigation.setOptions({ tabBarBadge: unreadCount > 0 ? unreadCount : undefined } as any);
  }, [unreadCount, navigation]);

  const loadUnreadCount = useCallback(async () => {
    try { const res = await MessageAPI.getUnreadCount(); setUnreadCount(res.count ?? 0); } catch { /* 静默 */ }
  }, [setUnreadCount]);

  useEffect(() => { loadUnreadCount(); }, [loadUnreadCount]);

  const loadMessages = useCallback(
    async (page: number, type: MessageType, isRefresh = false) => {
      if (loading) return;
      setLoading(true);
      try {
        const params: Record<string, any> = { pageNum: page, pageSize: PAGE_SIZE };
        if (type !== 'all') params.type = type;
        const res = await MessageAPI.getPage(params);
        const list = (res.list as unknown as MessageVO[]) ?? [];
        if (isRefresh || page === 1) setMessages(list);
        else setMessages(prev => [...prev, ...list]);
        setHasMore(list.length >= PAGE_SIZE);
        setPageNum(page);
      } catch { Alert.alert('加载失败', '无法加载消息列表'); }
      finally { setLoading(false); setRefreshing(false); }
    },
    [loading],
  );

  useEffect(() => {
    setMessages([]); setPageNum(1); setHasMore(true);
    loadMessages(1, activeType, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeType]);

  const handleRefresh = useCallback(() => {
    setRefreshing(true); loadMessages(1, activeType, true); loadUnreadCount();
  }, [activeType, loadMessages, loadUnreadCount]);

  const handleLoadMore = useCallback(() => {
    if (hasMore && !loading) loadMessages(pageNum + 1, activeType);
  }, [hasMore, loading, pageNum, activeType, loadMessages]);

  const handlePressMessage = useCallback(async (item: MessageVO) => {
    if (item.readStatus === 0) {
      try {
        await MessageAPI.markRead(item.id);
        decrementUnread(1);
        setMessages(prev => prev.map(m => (m.id === item.id ? { ...m, readStatus: 1 } : m)));
      } catch { /* 标记失败不阻塞跳转 */ }
    }
    navigation.navigate('MessageDetail', { messageId: item.id });
  }, [navigation, decrementUnread]);

  const handleMarkAllRead = useCallback(() => {
    Alert.alert('确认', '确定将所有消息标记为已读？', [
      { text: '取消', style: 'cancel' },
      {
        text: '确定',
        onPress: async () => {
          try {
            await MessageAPI.markAllRead(activeType !== 'all' ? activeType : undefined);
            setUnreadCount(0);
            setMessages(prev => prev.map(m => ({ ...m, readStatus: 1 })));
          } catch { Alert.alert('操作失败', '请稍后重试'); }
        },
      },
    ]);
  }, [activeType, setUnreadCount]);

  const handleSettings = useCallback(() => { navigation.navigate('Notify' as any); }, [navigation]);

  const formatTime = (timeStr?: string) => {
    if (!timeStr) return '';
    const d = new Date(timeStr);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60 * 1000) return '刚刚';
    if (diff < 60 * 60 * 1000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / 3600000)}小时前`;
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'system': return colors.status.info;
      case 'processing': return colors.status.success;
      case 'activity': return colors.status.warning;
      default: return colors.text.tertiary;
    }
  };

  const renderItem = ({ item }: { item: MessageVO }) => (
    <TouchableOpacity
      style={[styles.messageItem, item.readStatus === 0 && styles.messageUnread]}
      activeOpacity={0.7}
      onPress={() => handlePressMessage(item)}
    >
      <View style={styles.messageLeft}>
        {item.readStatus === 0 && <View style={styles.unreadDot} />}
        <View style={[styles.typeBadge, { backgroundColor: getTypeColor(item.type) + '20' }]}>
          <Text style={[styles.typeBadgeText, { color: getTypeColor(item.type) }]}>{item.typeLabel || item.type}</Text>
        </View>
      </View>
      <View style={styles.messageBody}>
        <View style={styles.messageHeader}>
          <Text style={styles.messageTitle} numberOfLines={1}>{item.title}</Text>
          <Text style={styles.messageTime}>{formatTime(item.createTime)}</Text>
        </View>
        {item.summary ? <Text style={styles.messageSummary} numberOfLines={2}>{item.summary}</Text> : null}
      </View>
    </TouchableOpacity>
  );

  const renderEmpty = () => {
    if (loading) return null;
    return (
      <View style={styles.empty}>
        <Ionicons name="mail-outline" size={48} color={colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无消息</Text>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>消息</Text>
          <View style={styles.headerActions}>
            <TouchableOpacity onPress={handleMarkAllRead} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Text style={styles.markAllReadText}>全部已读</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={handleSettings} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Ionicons name="settings-outline" size={20} color={colors.text.secondary} />
            </TouchableOpacity>
          </View>
        </View>
        <View style={styles.typeTabs}>
          {TYPE_TABS.map(tab => (
            <TouchableOpacity
              key={tab.key}
              style={[styles.typeTab, activeType === tab.key && styles.typeTabActive]}
              onPress={() => setActiveType(tab.key)}
              activeOpacity={0.7}
            >
              <Text style={[styles.typeTabText, activeType === tab.key && styles.typeTabTextActive]}>{tab.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
        <FlatList
          data={messages}
          renderItem={renderItem}
          keyExtractor={item => String(item.id)}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[colors.primary]} />}
          onEndReached={handleLoadMore}
          onEndReachedThreshold={0.3}
          ListEmptyComponent={renderEmpty}
          ListFooterComponent={loading && messages.length > 0 ? <ActivityIndicator style={styles.loadMore} color={colors.primary} /> : null}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background.secondary },
  container: { flex: 1 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  headerTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  markAllReadText: { fontSize: 13, color: colors.primary, fontWeight: '500' },
  typeTabs: { flexDirection: 'row', marginHorizontal: spacing.md, marginBottom: spacing.sm, backgroundColor: colors.background.tertiary, borderRadius: layout.borderRadius.md, padding: 2 },
  typeTab: { flex: 1, paddingVertical: spacing.sm, alignItems: 'center', borderRadius: layout.borderRadius.sm },
  typeTabActive: { backgroundColor: colors.background.primary, ...layout.shadows.sm },
  typeTabText: { fontSize: 13, fontWeight: '500', color: colors.text.secondary },
  typeTabTextActive: { color: colors.primary },
  listContent: { paddingHorizontal: spacing.md, paddingBottom: spacing.xxxl, flexGrow: 1 },
  messageItem: { flexDirection: 'row', padding: spacing.md, marginBottom: spacing.sm, backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.md, ...layout.shadows.sm },
  messageUnread: { borderLeftWidth: 3, borderLeftColor: colors.primary },
  messageLeft: { alignItems: 'center', marginRight: spacing.sm, gap: spacing.xs },
  unreadDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.primary },
  typeBadge: { paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: layout.borderRadius.sm },
  typeBadgeText: { fontSize: 10, fontWeight: '600' },
  messageBody: { flex: 1, gap: 4 },
  messageHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  messageTitle: { flex: 1, fontSize: 15, fontWeight: '600', color: colors.text.primary, marginRight: spacing.sm },
  messageTime: { fontSize: 11, color: colors.text.tertiary },
  messageSummary: { fontSize: 13, color: colors.text.secondary, lineHeight: 18 },
  empty: { flex: 1, justifyContent: 'center', alignItems: 'center', gap: spacing.md, paddingTop: spacing.huge },
  emptyText: { fontSize: 15, color: colors.text.tertiary },
  loadMore: { paddingVertical: spacing.lg },
});
