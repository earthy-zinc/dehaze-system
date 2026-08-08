/**
 * 系统设置 (L2)
 *
 * 缓存清理 / 通知入口 / 版本号 / 退出登录
 */
import React, { useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Linking,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Ionicons from 'react-native-vector-icons/Ionicons';
import type { IoniconName } from '@/components/Icon';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';
import { useAuthStore } from '@/store';

interface SettingRow {
  icon: IoniconName;
  title: string;
  onPress: () => void;
  destructive?: boolean;
}

const PersonalSettingsScreen: React.FC = () => {
  const navigation = useNavigation();
  const logout = useAuthStore(s => s.logout);

  const handleClearCache = useCallback(() => {
    Alert.alert('清理缓存', '确定要清理缓存吗？这不会影响您的登录状态。', [
      { text: '取消', style: 'cancel' },
      {
        text: '确定',
        onPress: async () => {
          try {
            const keys = await AsyncStorage.getAllKeys();
            const keepKeys = ['auth-storage', 'SESSION_KEY'];
            const toRemove = keys.filter(k => !keepKeys.includes(k));
            if (toRemove.length > 0) {
              await AsyncStorage.multiRemove(toRemove);
            }
            Alert.alert('清理完成', `已清理 ${toRemove.length} 项缓存`);
          } catch {
            Alert.alert('清理失败', '请稍后重试');
          }
        },
      },
    ]);
  }, []);

  const handleOpenPrivacy = useCallback(() => {
    Linking.openURL('https://dehaze.example.com/privacy').catch(() => {});
  }, []);

  const handleOpenTerms = useCallback(() => {
    Linking.openURL('https://dehaze.example.com/terms').catch(() => {});
  }, []);

  const handleLogout = useCallback(() => {
    Alert.alert('确认退出', '确定要退出登录吗？', [
      { text: '取消', style: 'cancel' },
      { text: '确定', style: 'destructive', onPress: () => logout() },
    ]);
  }, [logout]);

  const rows: SettingRow[] = [
    { icon: 'trash-outline', title: '清理缓存', onPress: handleClearCache },
    { icon: 'shield-checkmark-outline', title: '隐私政策', onPress: handleOpenPrivacy },
    { icon: 'document-text-outline', title: '用户协议', onPress: handleOpenTerms },
  ];

  return (
    <View style={styles.container}>
      <AppHeader title="系统设置" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.content}>
      {/* 设置项 */}
      <View style={styles.card}>
        {rows.map((row, idx) => (
          <React.Fragment key={row.title}>
            {idx > 0 && <View style={styles.divider} />}
            <TouchableOpacity
              style={styles.row}
              onPress={row.onPress}
              activeOpacity={0.6}
            >
              <Ionicons
                name={row.icon}
                size={20}
                color={row.destructive ? theme.colors.status.error : theme.colors.text.secondary}
              />
              <Text style={[styles.rowTitle, row.destructive && { color: theme.colors.status.error }]}>
                {row.title}
              </Text>
              <Ionicons name="chevron-forward" size={16} color={theme.colors.text.tertiary} />
            </TouchableOpacity>
          </React.Fragment>
        ))}
      </View>

      {/* 版本号 */}
      <View style={styles.versionWrap}>
        <Text style={styles.versionText}>图像去雾系统 v1.0</Text>
        <Text style={styles.versionText}>React Native 0.81</Text>
      </View>

      {/* 退出登录 */}
      <TouchableOpacity
        style={styles.logoutBtn}
        onPress={handleLogout}
        activeOpacity={0.7}
      >
        <Ionicons name="log-out-outline" size={20} color={theme.colors.status.error} />
        <Text style={styles.logoutText}>退出登录</Text>
      </TouchableOpacity>
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    ...theme.layout.shadows.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: theme.spacing.md,
    gap: theme.spacing.sm,
  },
  rowTitle: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: theme.colors.border.light,
    marginHorizontal: theme.spacing.md,
  },
  versionWrap: {
    alignItems: 'center',
    marginTop: theme.spacing.xl,
    gap: 4,
  },
  versionText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  logoutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    marginTop: theme.spacing.lg,
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
});

export default PersonalSettingsScreen;
