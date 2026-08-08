/**
 * 反馈评价 (L2)
 *
 * FeedbackAPI 我的反馈列表 + 提交反馈表单
 */
import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  RefreshControl,
  TextInput,
  Alert,
  Modal,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { FeedbackAPI } from 'dehaze-sdk-js';
import type { FeedbackPageVO, FeedbackType } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PAGE_SIZE = 20;

const FEEDBACK_STATUS_MAP: Record<string, { label: string; color: string }> = {
  PENDING: { label: '待处理', color: '#f59e0b' },
  PROCESSING: { label: '处理中', color: theme.colors.primary },
  RESOLVED: { label: '已解决', color: theme.colors.status.success },
  CLOSED: { label: '已关闭', color: theme.colors.text.tertiary },
};

const PersonalFeedbackScreen: React.FC = () => {
  const navigation = useNavigation();
  const [feedbacks, setFeedbacks] = useState<FeedbackPageVO[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [type, setType] = useState<FeedbackType>('bug');
  const [submitting, setSubmitting] = useState(false);

  const titleRef = useRef<TextInput>(null);

  const loadFeedbacks = useCallback(async (pageNum = 1, isRefresh = false) => {
    try {
      if (isRefresh) setRefreshing(true);
      else if (pageNum === 1) setLoading(true);
      const result = await FeedbackAPI.listMyFeedback({ pageNum, pageSize: PAGE_SIZE });
      const list = result.list || [];
      if (pageNum === 1) setFeedbacks(list);
      else setFeedbacks(prev => [...prev, ...list]);
      setHasMore(list.length >= PAGE_SIZE);
      setPage(pageNum);
    } catch {
      Alert.alert('加载失败', '获取反馈列表失败，请重试');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadFeedbacks(1);
    }, [loadFeedbacks]),
  );

  const handleRefresh = useCallback(() => loadFeedbacks(1, true), [loadFeedbacks]);
  const handleLoadMore = useCallback(() => {
    if (hasMore && !refreshing) loadFeedbacks(page + 1);
  }, [hasMore, refreshing, page, loadFeedbacks]);

  const handleSubmit = useCallback(async () => {
    if (!title.trim()) {
      Alert.alert('提示', '请输入反馈标题');
      return;
    }
    setSubmitting(true);
    try {
      await FeedbackAPI.createFeedback({ feedbackType: type, title: title.trim(), content: content.trim() });
      Alert.alert('提交成功', '感谢您的反馈！');
      setShowForm(false);
      setTitle('');
      setContent('');
      setType('bug');
      loadFeedbacks(1);
    } catch {
      Alert.alert('提交失败', '请稍后重试');
    } finally {
      setSubmitting(false);
    }
  }, [title, content, type, loadFeedbacks]);

  const renderItem = useCallback(({ item }: { item: FeedbackPageVO }) => {
    const statusInfo = FEEDBACK_STATUS_MAP[item.status] || { label: item.status, color: theme.colors.text.tertiary };
    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.fbTitle} numberOfLines={1}>{item.title}</Text>
          <View style={[styles.statusBadge, { backgroundColor: statusInfo.color + '20' }]}>
            <Text style={[styles.statusText, { color: statusInfo.color }]}>{statusInfo.label}</Text>
          </View>
        </View>
        {item.content ? (
          <Text style={styles.fbContent} numberOfLines={2}>{item.content}</Text>
        ) : null}
        <Text style={styles.fbTime}>
          {item.createTime ? new Date(item.createTime).toLocaleDateString('zh-CN') : ''}
        </Text>
      </View>
    );
  }, []);

  const renderEmpty = () =>
    !loading ? (
      <View style={styles.empty}>
        <Ionicons name="chatbox-outline" size={48} color={theme.colors.text.tertiary} />
        <Text style={styles.emptyText}>暂无反馈</Text>
      </View>
    ) : null;

  return (
    <View style={styles.container}>
      <AppHeader title="反馈评价" showBack onBackPress={() => navigation.goBack()} />
      <FlatList
        data={feedbacks}
        renderItem={renderItem}
        keyExtractor={item => String(item.id)}
        contentContainerStyle={styles.list}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} colors={[theme.colors.primary]} tintColor={theme.colors.primary} />
        }
        onEndReached={handleLoadMore}
        onEndReachedThreshold={0.5}
        ListEmptyComponent={renderEmpty}
      />

      {/* 提交反馈 FAB */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => setShowForm(true)}
        activeOpacity={0.8}
      >
        <Ionicons name="add" size={28} color="#fff" />
      </TouchableOpacity>

      {/* 提交反馈模态 */}
      <Modal visible={showForm} animationType="slide" presentationStyle="pageSheet">
        <View style={styles.formContainer}>
          <View style={styles.formHeader}>
            <TouchableOpacity onPress={() => setShowForm(false)}>
              <Text style={styles.cancelText}>取消</Text>
            </TouchableOpacity>
            <Text style={styles.formTitle}>提交反馈</Text>
            <TouchableOpacity onPress={handleSubmit} disabled={submitting}>
              <Text style={[styles.submitText, submitting && styles.submitTextDisabled]}>
                {submitting ? '提交中...' : '提交'}
              </Text>
            </TouchableOpacity>
          </View>

          {/* 类型选择 */}
          <View style={styles.typeRow}>
            {(['bug', 'suggestion', 'experience', 'complaint'] as FeedbackType[]).map(t => (
              <TouchableOpacity
                key={t}
                style={[styles.typeChip, type === t && styles.typeChipActive]}
                onPress={() => setType(t)}
              >
                <Text style={[styles.typeChipText, type === t && styles.typeChipTextActive]}>
                  {t === 'bug' ? '问题反馈' : t === 'suggestion' ? '功能建议' : t === 'experience' ? '体验反馈' : '投诉'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <TextInput
            ref={titleRef}
            style={styles.titleInput}
            placeholder="反馈标题（必填）"
            placeholderTextColor={theme.colors.text.tertiary}
            value={title}
            onChangeText={setTitle}
            maxLength={100}
          />
          <TextInput
            style={styles.contentInput}
            placeholder="详细描述您的问题或建议..."
            placeholderTextColor={theme.colors.text.tertiary}
            value={content}
            onChangeText={setContent}
            multiline
            textAlignVertical="top"
            maxLength={1000}
          />
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  list: { padding: theme.spacing.md, flexGrow: 1 },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  fbTitle: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginRight: theme.spacing.sm,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: theme.layout.borderRadius.sm,
  },
  statusText: {
    fontSize: theme.typography.sizes.tiny,
    fontWeight: theme.typography.weights.semibold,
  },
  fbContent: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    marginBottom: 8,
  },
  fbTime: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  empty: { alignItems: 'center', paddingVertical: theme.spacing.xxxl },
  emptyText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary, marginTop: theme.spacing.sm },
  fab: {
    position: 'absolute',
    right: theme.spacing.lg,
    bottom: theme.spacing.lg,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: theme.colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    ...theme.layout.shadows.lg,
  },
  // 表单
  formContainer: { flex: 1, padding: theme.spacing.md },
  formHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
    marginBottom: theme.spacing.lg,
  },
  cancelText: { fontSize: theme.typography.sizes.medium, color: theme.colors.text.secondary },
  formTitle: { fontSize: theme.typography.sizes.large, fontWeight: theme.typography.weights.bold, color: theme.colors.text.primary },
  submitText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.primary },
  submitTextDisabled: { opacity: 0.5 },
  typeRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.lg,
  },
  typeChip: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: theme.layout.borderRadius.md,
    alignItems: 'center',
    backgroundColor: theme.colors.background.tertiary,
  },
  typeChipActive: {
    backgroundColor: theme.colors.primaryLight,
  },
  typeChipText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  typeChipTextActive: {
    color: theme.colors.primary,
  },
  titleInput: {
    height: 48,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
    borderRadius: theme.layout.borderRadius.md,
    paddingHorizontal: theme.spacing.md,
    fontSize: theme.typography.sizes.medium,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
  },
  contentInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
    borderRadius: theme.layout.borderRadius.md,
    padding: theme.spacing.md,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    lineHeight: 22,
  },
});

export default PersonalFeedbackScreen;
