/**
 * 个人中心页面
 *
 * 展示当前登录用户信息、权限概览、最近处理记录，提供退出登录入口。
 * 用户信息来自 AuthContext，历史记录复用图像输入模块的 historyStorage。
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  RefreshControl,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';
import type { IoniconName } from '@/components/Icon';

import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { useAuth } from '@/store';
import { theme } from '@/theme';
import { historyStorage } from '@/pages/image-input/services/historyStorage';
import type { HistoryRecord } from '@/pages/image-input/types/imageInput';
import ImageLoader from '@/components/ImageLoader';
import { extractFilename } from '@/utils/url';

type Props = NativeStackScreenProps<RootStackParamList, 'Profile'>;

/** 权限概览最大展示条数 */
const MAX_PERMS_DISPLAY = 5;
/** 最近处理记录最大展示条数 */
const MAX_HISTORY_DISPLAY = 3;

/** 去除角色前缀（ROLE_） */
function formatRole(role: string): string {
  return role.replace(/^ROLE_/, '');
}

const ProfileScreen: React.FC<Props> = ({ navigation }) => {
  const { state, logout, refreshUserInfo } = useAuth();
  const userInfo = state.userInfo;

  const [recentHistory, setRecentHistory] = useState<HistoryRecord[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  /** 加载最近处理记录 */
  const loadRecentHistory = useCallback(async () => {
    try {
      const history = await historyStorage.getHistory();
      setRecentHistory(history.slice(0, MAX_HISTORY_DISPLAY));
    } catch {
      setRecentHistory([]);
    }
  }, []);

  // 首次进入：若用户信息缺失则拉取，同时加载历史
  useEffect(() => {
    if (!userInfo) {
      // refreshUserInfo 失败时不阻塞页面渲染，用户可下拉刷新重试
      refreshUserInfo().catch(() => { /* 静默忽略：见上方注释 */ });
    }
    loadRecentHistory();
  }, [userInfo, refreshUserInfo, loadRecentHistory]);

  /** 下拉刷新 */
  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.allSettled([refreshUserInfo(), loadRecentHistory()]);
    setRefreshing(false);
  }, [refreshUserInfo, loadRecentHistory]);

  /** 退出登录二次确认 */
  const handleLogout = useCallback(() => {
    Alert.alert('确认退出', '确定要退出登录吗？', [
      { text: '取消', style: 'cancel' },
      {
        text: '确定',
        style: 'destructive',
        onPress: () => {
          logout();
        },
      },
    ]);
  }, [logout]);

  /** 跳转图像输入页查看完整历史（直接切换到历史 Tab） */
  const handleViewAllHistory = useCallback(() => {
    navigation.navigate('ImageInput', { initialMethod: 'history' });
  }, [navigation]);

  const nickname = userInfo?.nickname || '未登录';
  const username = userInfo?.username || '—';
  const avatarLetter = nickname.charAt(0).toUpperCase();
  const roles = userInfo?.roles ?? [];
  const permissions = userInfo?.permissions ?? [];
  const displayPerms = permissions.slice(0, MAX_PERMS_DISPLAY);
  const remainingPerms = Math.max(0, permissions.length - MAX_PERMS_DISPLAY);

  return (
    <MainLayout title="个人中心" showBack showBottomNav={false}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={[theme.colors.primary]}
            tintColor={theme.colors.primary}
          />
        }
      >
        {/* 用户信息 Hero */}
        <LinearGradient
          colors={[theme.colors.primary, '#6366f1']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <View style={styles.avatarWrap}>
            <Text style={styles.avatarText}>{avatarLetter}</Text>
          </View>
          <Text style={styles.nickname} numberOfLines={1}>
            {nickname}
          </Text>
          <Text style={styles.username} numberOfLines={1}>
            @{username}
          </Text>
          {roles.length > 0 && (
            <View style={styles.roleTags}>
              {roles.map(role => (
                <View key={role} style={styles.roleTag}>
                  <Text style={styles.roleTagText}>{formatRole(role)}</Text>
                </View>
              ))}
            </View>
          )}
        </LinearGradient>

        {/* 账号信息 */}
        <SectionWrap icon="information-circle" title="账号信息">
          <View style={styles.card}>
            <InfoRow
              label="用户ID"
              value={userInfo ? String(userInfo.userId) : '—'}
            />
            <InfoRow label="用户名" value={username} />
            <InfoRow label="昵称" value={nickname} last />
          </View>
        </SectionWrap>

        {/* 权限概览 */}
        <SectionWrap
          icon="lock-closed-outline"
          title="权限概览"
          extra={`${permissions.length} 项`}
        >
          <View style={styles.card}>
            {displayPerms.length > 0 ? (
              <View style={styles.permList}>
                {displayPerms.map(perm => (
                  <View key={perm} style={styles.permChip}>
                    <Ionicons
                      name="key-outline"
                      size={12}
                      color={theme.colors.primary}
                      style={styles.permIcon}
                    />
                    <Text style={styles.permText} numberOfLines={1}>
                      {perm}
                    </Text>
                  </View>
                ))}
                {remainingPerms > 0 && (
                  <View style={[styles.permChip, styles.permChipMore]}>
                    <Text style={styles.permMoreText}>+{remainingPerms} 项</Text>
                  </View>
                )}
              </View>
            ) : (
              <Text style={styles.emptyText}>暂无权限信息</Text>
            )}
          </View>
        </SectionWrap>

        {/* 最近处理记录 */}
        <SectionWrap
          icon="time-outline"
          title="最近处理"
          extra="查看全部"
          onExtraPress={handleViewAllHistory}
        >
          <View style={styles.card}>
            {recentHistory.length > 0 ? (
              recentHistory.map((record, index) => (
                <HistoryRow
                  key={record.id}
                  record={record}
                  last={index === recentHistory.length - 1}
                />
              ))
            ) : (
              <Text style={styles.emptyText}>暂无处理记录</Text>
            )}
          </View>
        </SectionWrap>

        {/* 退出登录 */}
        <TouchableOpacity
          style={styles.logoutButton}
          onPress={handleLogout}
          activeOpacity={0.7}
        >
          <Ionicons name="log-out-outline" size={20} color={theme.colors.status.error} />
          <Text style={styles.logoutText}>退出登录</Text>
        </TouchableOpacity>
      </ScrollView>
    </MainLayout>
  );
};

/** 区块包装 */
const SectionWrap: React.FC<{
  icon: string;
  title: string;
  extra?: string;
  onExtraPress?: () => void;
  children: React.ReactNode;
}> = ({ icon, title, extra, onExtraPress, children }) => (
  <View style={styles.sectionWrap}>
    <View style={styles.sectionTitleRow}>
      <View style={styles.sectionTitleIcon}>
        <Ionicons name={icon as IoniconName} size={16} color={theme.colors.primary} />
      </View>
      <Text style={styles.sectionTitleText}>{title}</Text>
      {extra && (
        <TouchableOpacity
          onPress={onExtraPress}
          hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
        >
          <Text style={styles.sectionExtra}>{extra}</Text>
        </TouchableOpacity>
      )}
    </View>
    {children}
  </View>
);

/** 信息行 */
const InfoRow: React.FC<{ label: string; value: string; last?: boolean }> = ({
  label,
  value,
  last,
}) => (
  <View style={[styles.infoRow, last && styles.infoRowLast]}>
    <Text style={styles.infoLabel}>{label}</Text>
    <Text style={styles.infoValue} numberOfLines={1}>
      {value}
    </Text>
  </View>
);

/** 历史记录紧凑行 */
const HistoryRow: React.FC<{ record: HistoryRecord; last?: boolean }> = ({
  record,
  last,
}) => {
  const filename = extractFilename(record.originalImageUrl);
  const time = historyStorage.formatTimestamp(record.createTime);
  const thumbUrl = record.originalThumbnailUrl || '';

  return (
    <View style={[styles.historyRow, last && styles.historyRowLast]}>
      <View style={styles.historyThumb}>
        {thumbUrl ? (
          <ImageLoader
            source={{ uri: thumbUrl }}
            style={styles.historyThumbImage}
            resizeMode="cover"
          />
        ) : (
          <Ionicons name="image-outline" size={20} color={theme.colors.text.tertiary} />
        )}
      </View>
      <View style={styles.historyInfo}>
        <Text style={styles.historyFilename} numberOfLines={1}>
          {filename}
        </Text>
        <View style={styles.historyMeta}>
          {record.algorithmName && (
            <Text style={styles.historyAlgo} numberOfLines={1}>
              {record.algorithmName}
            </Text>
          )}
          {time && <Text style={styles.historyTime}>· {time}</Text>}
        </View>
      </View>
      <Ionicons
        name="chevron-forward"
        size={16}
        color={theme.colors.text.tertiary}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: theme.spacing.xxxl,
  },
  // Hero
  hero: {
    alignItems: 'center',
    paddingVertical: theme.spacing.xl,
    paddingHorizontal: theme.spacing.lg,
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.xxl,
    ...theme.layout.shadows.lg,
  },
  avatarWrap: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  avatarText: {
    fontSize: 34,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  nickname: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
    letterSpacing: theme.typography.letterSpacing.normal,
  },
  username: {
    fontSize: theme.typography.sizes.bodySmall,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 4,
  },
  roleTags: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 6,
    marginTop: theme.spacing.sm,
  },
  roleTag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  roleTagText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  // 区块
  sectionWrap: {
    marginTop: theme.spacing.lg,
    paddingHorizontal: theme.spacing.md,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  sectionTitleIcon: {
    width: 26,
    height: 26,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: `${theme.colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sectionTitleText: {
    flex: 1,
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    letterSpacing: 0.3,
  },
  sectionExtra: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  // 卡片
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  // 信息行
  infoRow: {
    flexDirection: 'row',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.colors.border.light,
  },
  infoRowLast: {
    borderBottomWidth: 0,
  },
  infoLabel: {
    width: 80,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    fontWeight: theme.typography.weights.medium,
  },
  infoValue: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    fontWeight: theme.typography.weights.medium,
  },
  // 权限
  permList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  permChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.primaryLight,
    maxWidth: '100%',
  },
  permIcon: {
    marginRight: 4,
  },
  permText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  permChipMore: {
    backgroundColor: theme.colors.background.tertiary,
  },
  permMoreText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.semibold,
  },
  // 空状态
  emptyText: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    textAlign: 'center',
    paddingVertical: theme.spacing.md,
  },
  // 历史记录行
  historyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: theme.colors.border.light,
    gap: theme.spacing.sm,
  },
  historyRowLast: {
    borderBottomWidth: 0,
  },
  historyThumb: {
    width: 40,
    height: 40,
    borderRadius: theme.layout.borderRadius.sm,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  historyThumbImage: {
    width: '100%',
    height: '100%',
  },
  historyInfo: {
    flex: 1,
  },
  historyFilename: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
    marginBottom: 2,
  },
  historyMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  historyAlgo: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
    flexShrink: 1,
  },
  historyTime: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  // 退出登录
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.xl,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.primary,
    ...theme.layout.shadows.sm,
  },
  logoutText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.status.error,
  },
});

export default ProfileScreen;
