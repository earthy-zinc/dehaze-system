/**
 * 个人中心（我的 Tab）
 *
 * 布局：用户卡 + VIP 横幅 + 数据统计 + 四组入口 + 管理入口（权限过滤）+ 退出登录
 * 用户信息来自 useAuthStore，会员/额度/收藏统计数据异步加载
 */
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  RefreshControl,
  Image,
} from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import LinearGradient from 'react-native-linear-gradient';
import Ionicons from 'react-native-vector-icons/Ionicons';
import type { IoniconName } from '@/components/Icon';
import { MemberAPI, ModelAPI, FavoriteAPI, TaskAPI } from 'dehaze-sdk-js';
import type { MemberProfileVO } from 'dehaze-sdk-js';

import type { ProfileStackParamList } from '@/routes/types';
import { useAuthStore } from '@/store';
import { theme } from '@/theme';

type Props = NativeStackScreenProps<ProfileStackParamList, 'Profile'>;

// ==================== 类型 ====================

interface StatItem {
  label: string;
  value: string;
  route: keyof ProfileStackParamList;
}

interface EntryItem {
  icon: IoniconName;
  title: string;
  route: keyof ProfileStackParamList;
  permission?: string;
}

interface EntryGroup {
  title: string;
  entries: EntryItem[];
}

// ==================== 入口配置 ====================

const PERSONAL_DATA_ENTRIES: EntryItem[] = [
  { icon: 'document-text-outline', title: '我的文件', route: 'PersonalFiles' },
  { icon: 'server-outline', title: '我的数据集', route: 'Dataset' },
  { icon: 'time-outline', title: '处理历史', route: 'Task' },
  { icon: 'heart-outline', title: '我的收藏', route: 'PersonalFavorites' },
];

const BUSINESS_ENTRIES: EntryItem[] = [
  { icon: 'diamond-outline', title: '我的会员', route: 'PersonalMember' },
  { icon: 'cube-outline', title: '我的套餐', route: 'PersonalPackage' },
  { icon: 'cart-outline', title: '我的订单', route: 'PersonalOrders' },
  { icon: 'wallet-outline', title: '我的额度', route: 'PersonalQuota' },
  { icon: 'chatbox-outline', title: '反馈评价', route: 'PersonalFeedback' },
];

const OTHER_ENTRIES: EntryItem[] = [
  { icon: 'settings-outline', title: '系统设置', route: 'PersonalSettings' },
  { icon: 'help-circle-outline', title: '帮助中心', route: 'PersonalHelp' },
  { icon: 'information-circle-outline', title: '关于我们', route: 'PersonalAbout' },
  { icon: 'notifications-outline', title: '消息设置', route: 'Notify' },
];

// 管理入口分组（权限过滤，与 dev-admin 对齐）
const ADMIN_GROUPS: EntryGroup[] = [
  {
    title: '工作台',
    entries: [
      { icon: 'speedometer-outline', title: '工作台', route: 'SystemDashboard' },
    ],
  },
  {
    title: '算法与数据',
    entries: [
      { icon: 'git-network-outline', title: '算法管理', route: 'SystemAlgorithm', permission: 'sys:algorithm:*' },
      { icon: 'server-outline', title: '数据集管理', route: 'SystemDataset', permission: 'sys:dataset:*' },
    ],
  },
  {
    title: '系统管理',
    entries: [
      { icon: 'people-outline', title: '用户管理', route: 'SystemUser', permission: 'sys:user:*' },
      { icon: 'shield-checkmark-outline', title: '角色管理', route: 'SystemRole', permission: 'sys:role:*' },
      { icon: 'list-outline', title: '菜单管理', route: 'SystemMenu', permission: 'sys:menu:*' },
      { icon: 'business-outline', title: '部门管理', route: 'SystemDept', permission: 'sys:dept:*' },
      { icon: 'book-outline', title: '字典管理', route: 'SystemDict', permission: 'sys:dict:*' },
      { icon: 'timer-outline', title: '任务管理', route: 'SystemTask', permission: 'sys:task:*' },
    ],
  },
  {
    title: '运营管理',
    entries: [
      { icon: 'diamond-outline', title: '会员管理', route: 'SystemMember', permission: 'sys:member:*' },
      { icon: 'cube-outline', title: '套餐管理', route: 'SystemPackage', permission: 'sys:package:*' },
      { icon: 'cart-outline', title: '订单管理', route: 'SystemOrder', permission: 'sys:order:*' },
      { icon: 'chatbox-outline', title: '反馈评价管理', route: 'SystemFeedback', permission: 'sys:feedback:*' },
      { icon: 'trending-up-outline', title: '推荐管理', route: 'SystemRecommend', permission: 'sys:recommendation:*' },
      { icon: 'notifications-outline', title: '消息管理', route: 'SystemMessage', permission: 'sys:notify:*' },
    ],
  },
];

// ==================== 组件 ====================

const ProfileScreen: React.FC<Props> = ({ navigation }) => {
  const userInfo = useAuthStore(s => s.userInfo);
  const logout = useAuthStore(s => s.logout);
  const refreshUserInfo = useAuthStore(s => s.refreshUserInfo);
  const hasPerm = useAuthStore(s => s.hasPerm);

  const [member, setMember] = useState<MemberProfileVO | null>(null);
  const [quotaRemaining, setQuotaRemaining] = useState<number | null>(null);
  const [favoriteCount, setFavoriteCount] = useState<number | null>(null);
  const [taskCount, setTaskCount] = useState<number | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // 加载会员信息
  const loadMember = useCallback(async () => {
    try {
      const profile = await MemberAPI.getProfile();
      setMember(profile);
    } catch {
      setMember(null);
    }
  }, []);

  // 加载统计数据
  const loadStats = useCallback(async () => {
    try {
      const quota = await ModelAPI.getQuota();
      setQuotaRemaining(quota.remaining ?? null);
    } catch {
      setQuotaRemaining(null);
    }
    try {
      const fav = await FavoriteAPI.getCount();
      if (fav && fav.length > 0) {
        const total = fav.reduce((sum, item) => sum + (item.count || 0), 0);
        setFavoriteCount(total);
      }
    } catch {
      setFavoriteCount(null);
    }
    try {
      const tasks = await TaskAPI.getPage({ pageNum: 1, pageSize: 1 });
      setTaskCount(tasks.total ?? null);
    } catch {
      setTaskCount(null);
    }
  }, []);

  useEffect(() => {
    loadMember();
    loadStats();
  }, [loadMember, loadStats]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.allSettled([refreshUserInfo(), loadMember(), loadStats()]);
    setRefreshing(false);
  }, [refreshUserInfo, loadMember, loadStats]);

  // 退出登录
  const handleLogout = useCallback(() => {
    Alert.alert('确认退出', '确定要退出登录吗？', [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: () => logout() },
    ]);
  }, [logout]);

  // 过滤管理入口
  const visibleAdminGroups = useMemo(
    () =>
      ADMIN_GROUPS.map(g => ({
        ...g,
        entries: g.entries.filter(e => (e.permission ? hasPerm(e.permission) : true)),
      })).filter(g => g.entries.length > 0),
    [hasPerm],
  );

  // 数据统计
  const stats: StatItem[] = useMemo(
    () => [
      { label: '剩余额度', value: quotaRemaining !== null ? `${quotaRemaining} 次` : '-', route: 'PersonalQuota' },
      { label: '处理次数', value: taskCount !== null ? `${taskCount}` : '-', route: 'Task' },
      { label: '我的收藏', value: favoriteCount !== null ? `${favoriteCount}` : '-', route: 'PersonalFavorites' },
    ],
    [quotaRemaining, taskCount, favoriteCount],
  );

  // VIP 判断
  const isVip = !!member && member.levelCode !== 'level_0' && member.levelCode !== 'level_1';
  const avatarLetter = (userInfo?.nickname || userInfo?.username || 'U').charAt(0).toUpperCase();
  const nickname = userInfo?.nickname || '未登录';
  const roles = userInfo?.roles ?? [];

  const navigateTo = useCallback(
    (route: keyof ProfileStackParamList) => {
      (navigation.navigate as (screen: string) => void)(route as string);
    },
    [navigation],
  );

  // ==================== 渲染 ====================

  const renderEntryRow = (entry: EntryItem) => (
    <TouchableOpacity
      key={String(entry.route)}
      style={styles.entryRow}
      onPress={() => navigateTo(entry.route)}
      activeOpacity={0.6}
    >
      <View style={styles.entryIconWrap}>
        <Ionicons name={entry.icon} size={20} color={theme.colors.text.secondary} />
      </View>
      <Text style={styles.entryTitle}>{entry.title}</Text>
      <Ionicons name="chevron-forward" size={16} color={theme.colors.text.tertiary} />
    </TouchableOpacity>
  );

  const renderGroup = (group: EntryGroup) => (
    <View key={group.title} style={styles.groupSection}>
      <Text style={styles.groupTitle}>{group.title}</Text>
      <View style={styles.groupCard}>
        {group.entries.map((entry, idx) => (
          <React.Fragment key={String(entry.route)}>
            {idx > 0 && <View style={styles.entryDivider} />}
            {renderEntryRow(entry)}
          </React.Fragment>
        ))}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
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
        {/* 用户卡 */}
        <LinearGradient
          colors={[theme.colors.primary, '#6366f1']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.userCard}
        >
          <View style={styles.userCardInner}>
            <View style={styles.avatarWrap}>
              {userInfo?.avatar ? (
                <Image source={{ uri: userInfo.avatar }} style={styles.avatarImg} />
              ) : (
                <Text style={styles.avatarLetter}>{avatarLetter}</Text>
              )}
            </View>
            <View style={styles.userInfo}>
              <Text style={styles.nickname} numberOfLines={1}>{nickname}</Text>
              <View style={styles.roleRow}>
                {roles.length > 0 ? (
                  roles.slice(0, 2).map(role => (
                    <View key={role} style={styles.roleTag}>
                      <Text style={styles.roleTagText}>
                        {role.replace(/^ROLE_/, '')}
                      </Text>
                    </View>
                  ))
                ) : (
                  <Text style={styles.noRoleText}>普通用户</Text>
                )}
              </View>
            </View>
          </View>
        </LinearGradient>

        {/* VIP 横幅 */}
        <TouchableOpacity
          style={styles.vipBanner}
          onPress={() => navigateTo('PersonalMember')}
          activeOpacity={0.8}
        >
          <View style={styles.vipLeft}>
            <Ionicons name="diamond" size={22} color="#f59e0b" />
            <View style={styles.vipTextWrap}>
              <Text style={styles.vipTitle}>
                {isVip ? `${member?.levelName || '会员'}专属权益` : '开通 VIP 畅享更多次数'}
              </Text>
              <Text style={styles.vipDesc}>
                {isVip
                  ? `成长值 ${member?.growthValue || 0}，点击查看详情`
                  : '解锁全部高级功能'}
              </Text>
            </View>
          </View>
          <View style={styles.vipAction}>
            <Text style={styles.vipActionText}>{isVip ? '详情' : '去开通'}</Text>
            <Ionicons name="chevron-forward" size={14} color={theme.colors.primary} />
          </View>
        </TouchableOpacity>

        {/* 数据统计 */}
        <View style={styles.statsCard}>
          {stats.map((stat, idx) => (
            <TouchableOpacity
              key={stat.label}
              style={[styles.statItem, idx < stats.length - 1 && styles.statDivider]}
              onPress={() => navigateTo(stat.route)}
              activeOpacity={0.6}
            >
              <Text style={styles.statValue}>{stat.value}</Text>
              <Text style={styles.statLabel}>{stat.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* 个人数据分组 */}
        <View style={styles.groupSection}>
          <Text style={styles.groupTitle}>个人数据</Text>
          <View style={styles.groupCard}>
            {PERSONAL_DATA_ENTRIES.map((entry, idx) => (
              <React.Fragment key={String(entry.route)}>
                {idx > 0 && <View style={styles.entryDivider} />}
                {renderEntryRow(entry)}
              </React.Fragment>
            ))}
          </View>
        </View>

        {/* 商业服务分组 */}
        <View style={styles.groupSection}>
          <Text style={styles.groupTitle}>商业服务</Text>
          <View style={styles.groupCard}>
            {BUSINESS_ENTRIES.map((entry, idx) => (
              <React.Fragment key={String(entry.route)}>
                {idx > 0 && <View style={styles.entryDivider} />}
                {renderEntryRow(entry)}
              </React.Fragment>
            ))}
          </View>
        </View>

        {/* 其他分组 */}
        <View style={styles.groupSection}>
          <Text style={styles.groupTitle}>其他</Text>
          <View style={styles.groupCard}>
            {OTHER_ENTRIES.map((entry, idx) => (
              <React.Fragment key={String(entry.route)}>
                {idx > 0 && <View style={styles.entryDivider} />}
                {renderEntryRow(entry)}
              </React.Fragment>
            ))}
          </View>
        </View>

        {/* 管理入口分组（权限过滤） */}
        {visibleAdminGroups.map(renderGroup)}

        {/* 退出登录 */}
        <TouchableOpacity
          style={styles.logoutBtn}
          onPress={handleLogout}
          activeOpacity={0.7}
        >
          <Ionicons name="log-out-outline" size={20} color={theme.colors.status.error} />
          <Text style={styles.logoutText}>退出登录</Text>
        </TouchableOpacity>

        {/* 页脚 */}
        <Text style={styles.footer}>图像去雾系统 v1.0</Text>
      </ScrollView>
    </View>
  );
};

// ==================== 样式 ====================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: theme.spacing.xxxl,
  },
  // 用户卡
  userCard: {
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.xxl,
    ...theme.layout.shadows.lg,
  },
  userCardInner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: theme.spacing.lg,
    gap: theme.spacing.md,
  },
  avatarWrap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(255,255,255,0.25)',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  avatarImg: {
    width: '100%',
    height: '100%',
  },
  avatarLetter: {
    fontSize: 28,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  userInfo: {
    flex: 1,
  },
  nickname: {
    fontSize: theme.typography.sizes.large,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  roleRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 6,
  },
  roleTag: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  roleTagText: {
    fontSize: theme.typography.sizes.tiny,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  noRoleText: {
    fontSize: theme.typography.sizes.tiny,
    color: 'rgba(255,255,255,0.7)',
  },
  // VIP 横幅
  vipBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    padding: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: '#fffbeb',
    borderWidth: 1,
    borderColor: '#fde68a',
  },
  vipLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    flex: 1,
  },
  vipTextWrap: {
    flex: 1,
  },
  vipTitle: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: '#92400e',
  },
  vipDesc: {
    fontSize: theme.typography.sizes.small,
    color: '#a16207',
    marginTop: 2,
  },
  vipAction: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
  },
  vipActionText: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.primary,
  },
  // 数据统计
  statsCard: {
    flexDirection: 'row',
    marginHorizontal: theme.spacing.md,
    marginTop: theme.spacing.md,
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    ...theme.layout.shadows.sm,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
  },
  statDivider: {
    borderRightWidth: StyleSheet.hairlineWidth,
    borderRightColor: theme.colors.border.light,
  },
  statValue: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  statLabel: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 4,
  },
  // 分组
  groupSection: {
    marginTop: theme.spacing.lg,
    paddingHorizontal: theme.spacing.md,
  },
  groupTitle: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  groupCard: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    ...theme.layout.shadows.sm,
  },
  // 入口行
  entryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.md,
    gap: theme.spacing.sm,
  },
  entryIconWrap: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  entryTitle: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  entryDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: theme.colors.border.light,
    marginHorizontal: theme.spacing.md,
  },
  // 退出登录
  logoutBtn: {
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
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.status.error,
  },
  // 页脚
  footer: {
    textAlign: 'center',
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.lg,
    paddingBottom: theme.spacing.md,
  },
});

export default ProfileScreen;
