/**
 * 消息详情页 (L2)
 *
 * 对接 MessageAPI.getDetail，rich-text 内容渲染。
 * 按 05-菜单与页面层级规划 2.4 节设计。
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NavigationProp } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { MessageAPI } from 'dehaze-sdk-js';
import type { MessageVO } from 'dehaze-sdk-js';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import { useMessagesStore } from '@/store/messages';
import type { RootStackParamList } from '@/routes/types';

type MessageDetailRouteProp = RouteProp<RootStackParamList, 'MessageDetail'>;

const MessageDetailScreen: React.FC = () => {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();
  const route = useRoute<MessageDetailRouteProp>();
  const { messageId } = route.params;
  const { decrementUnread } = useMessagesStore();

  const [message, setMessage] = useState<MessageVO | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDetail = useCallback(async () => {
    setLoading(true);
    try {
      const data = await MessageAPI.getDetail(messageId);
      setMessage(data);
      // 如果未读则标记已读
      if (data.readStatus === 0) {
        try {
          await MessageAPI.markRead(messageId);
          decrementUnread(1);
        } catch {
          // 静默
        }
      }
    } catch {
      Alert.alert('加载失败', '无法加载消息详情', [
        { text: '返回', onPress: () => navigation.goBack() },
      ]);
    } finally {
      setLoading(false);
    }
  }, [messageId, navigation, decrementUnread]);

  useEffect(() => {
    loadDetail();
  }, [loadDetail]);

  const formatTime = (timeStr?: string) => {
    if (!timeStr) return '';
    const d = new Date(timeStr);
    return `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  if (!message) {
    return (
      <View style={styles.centered}>
        <Ionicons name="alert-circle-outline" size={48} color={colors.text.tertiary} />
        <Text style={styles.errorText}>消息不存在或已被删除</Text>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Text style={styles.backBtnText}>返回</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 自定义导航栏 (L2: 返回 + 标题) */}
      <View style={styles.navbar}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.navTitle} numberOfLines={1}>
          消息详情
        </Text>
        <View style={styles.navPlaceholder} />
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* 消息头部 */}
        <View style={styles.detailHeader}>
          <View style={styles.typeRow}>
            <View style={[styles.typeBadge, { backgroundColor: colors.primaryLight }]}>
              <Text style={[styles.typeBadgeText, { color: colors.primary }]}>
                {message.typeLabel || message.type}
              </Text>
            </View>
            {message.priority > 0 && (
              <View style={styles.priorityBadge}>
                <Ionicons name="flag" size={12} color={colors.status.warning} />
                <Text style={styles.priorityText}>重要</Text>
              </View>
            )}
          </View>
          <Text style={styles.detailTitle}>{message.title}</Text>
          <Text style={styles.detailTime}>{formatTime(message.createTime)}</Text>
        </View>

        {/* 消息正文 */}
        <View style={styles.contentCard}>
          <Text style={styles.contentText}>
            {message.content || message.summary || '暂无详细内容'}
          </Text>
        </View>

        {/* 跳转链接（如有） */}
        {message.jumpUrl && (
          <TouchableOpacity
            style={styles.jumpBtn}
            activeOpacity={0.7}
            onPress={() => {
              Alert.alert('提示', '跳转链接功能开发中');
            }}
          >
            <Ionicons name="open-outline" size={16} color={colors.primary} />
            <Text style={styles.jumpBtnText}>查看详情</Text>
          </TouchableOpacity>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.secondary,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.background.secondary,
  },
  errorText: {
    fontSize: 15,
    color: colors.text.secondary,
  },
  backBtn: {
    marginTop: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: layout.borderRadius.md,
    backgroundColor: colors.primary,
  },
  backBtnText: {
    color: '#fff',
    fontWeight: '600',
  },
  navbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background.primary,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border.light,
  },
  navTitle: {
    flex: 1,
    textAlign: 'center',
    fontSize: 17,
    fontWeight: '600',
    color: colors.text.primary,
  },
  navPlaceholder: {
    width: 32,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.md,
    paddingBottom: spacing.xxxl,
  },
  detailHeader: {
    marginBottom: spacing.lg,
  },
  typeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  typeBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: layout.borderRadius.sm,
  },
  typeBadgeText: {
    fontSize: 11,
    fontWeight: '600',
  },
  priorityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: layout.borderRadius.sm,
    backgroundColor: '#fef3c7',
  },
  priorityText: {
    fontSize: 11,
    color: colors.status.warning,
    fontWeight: '500',
  },
  detailTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: colors.text.primary,
    lineHeight: 28,
    marginBottom: spacing.xs,
  },
  detailTime: {
    fontSize: 13,
    color: colors.text.tertiary,
  },
  contentCard: {
    backgroundColor: colors.background.primary,
    borderRadius: layout.borderRadius.lg,
    padding: spacing.lg,
    ...layout.shadows.sm,
  },
  contentText: {
    fontSize: 15,
    color: colors.text.primary,
    lineHeight: 24,
  },
  jumpBtn: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: layout.borderRadius.md,
    backgroundColor: colors.primaryLight,
  },
  jumpBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary,
  },
});

export default MessageDetailScreen;
