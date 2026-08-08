/**
 * 消息设置 (L2)
 *
 * NotificationSettingAPI.get/update 通知开关/免打扰时段/处理完成提醒
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Switch,
  Alert,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NotificationSettingAPI } from 'dehaze-sdk-js';
import type { NotificationSettings, NotificationSettingsForm } from 'dehaze-sdk-js';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PersonalNotifyScreen: React.FC = () => {
  const navigation = useNavigation();
  const [settings, setSettings] = useState<NotificationSettings | null>(null);
  const [loading, setLoading] = useState(true);

  const loadSettings = useCallback(async () => {
    try {
      const s = await NotificationSettingAPI.get();
      setSettings(s);
    } catch {
      setSettings(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const updateSetting = useCallback(
    async (partial: NotificationSettingsForm) => {
      try {
        await NotificationSettingAPI.update(partial);
        setSettings(prev => (prev ? { ...prev, ...partial } as NotificationSettings : prev));
      } catch {
        Alert.alert('保存失败', '请稍后重试');
      }
    },
    [],
  );

  const togglePush = useCallback(
    (value: boolean) => {
      updateSetting({ pushEnabled: value });
    },
    [updateSetting],
  );

  const toggleDnd = useCallback(
    (value: boolean) => {
      updateSetting({ dndEnabled: value });
    },
    [updateSetting],
  );

  const toggleTypeChannel = useCallback(
    (type: string, value: boolean) => {
      const current = settings?.preferences?.typeChannels ?? {};
      updateSetting({
        preferences: {
          ...settings?.preferences,
          typeChannels: { ...current, [type]: { push: value } },
        },
      });
    },
    [settings, updateSetting],
  );

  if (loading) {
    return (
      <View style={styles.container}>
        <AppHeader title="通知设置" showBack onBackPress={() => navigation.goBack()} />
        <View style={styles.loadingWrap}>
          <Text style={styles.loadingText}>加载中...</Text>
        </View>
      </View>
    );
  }

  const pushEnabled = settings?.pushEnabled ?? true;
  const dndEnabled = settings?.dndEnabled ?? false;
  const dndStart = settings?.dndStart || '22:00';
  const dndEnd = settings?.dndEnd || '07:00';
  const typeChannels = settings?.preferences?.typeChannels ?? {};

  return (
    <View style={styles.container}>
      <AppHeader title="通知设置" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.content}>
      {/* 推送通知总开关 */}
      <View style={styles.card}>
        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Ionicons name="notifications-outline" size={20} color={theme.colors.text.secondary} />
            <Text style={styles.switchLabel}>推送通知</Text>
          </View>
          <Switch
            value={pushEnabled}
            onValueChange={togglePush}
            trackColor={{ false: theme.colors.background.tertiary, true: theme.colors.primaryLight }}
            thumbColor={pushEnabled ? theme.colors.primary : theme.colors.text.tertiary}
          />
        </View>
        <Text style={styles.switchDesc}>关闭后将不会收到任何推送通知</Text>
      </View>

      {/* 免打扰时段 */}
      <View style={styles.card}>
        <View style={styles.switchRow}>
          <View style={styles.switchInfo}>
            <Ionicons name="moon-outline" size={20} color={theme.colors.text.secondary} />
            <Text style={styles.switchLabel}>免打扰时段</Text>
          </View>
          <Switch
            value={dndEnabled}
            onValueChange={toggleDnd}
            trackColor={{ false: theme.colors.background.tertiary, true: theme.colors.primaryLight }}
            thumbColor={dndEnabled ? theme.colors.primary : theme.colors.text.tertiary}
          />
        </View>
        <Text style={styles.switchDesc}>
          开启后，{dndStart} 至 {dndEnd} 期间不推送通知
        </Text>
      </View>

      {/* 通知类型开关 */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>通知类型</Text>
        <Text style={styles.sectionDesc}>选择您希望接收的通知类型</Text>
      </View>
      <View style={styles.card}>
        {([
          { key: 'system', label: '系统通知', desc: '系统维护、更新等通知', icon: 'settings-outline' as const },
          { key: 'process', label: '处理完成提醒', desc: '去雾/评估任务完成时通知', icon: 'checkmark-circle-outline' as const },
          { key: 'activity', label: '活动通知', desc: '优惠活动、新功能上线等', icon: 'sparkles-outline' as const },
        ]).map((item, idx) => {
          const enabled = typeChannels[item.key]?.push !== false;
          return (
            <React.Fragment key={item.key}>
              {idx > 0 && <View style={styles.divider} />}
              <View style={styles.switchRow}>
                <View style={styles.switchInfo}>
                  <Ionicons name={item.icon} size={20} color={theme.colors.text.secondary} />
                  <View style={styles.switchTextWrap}>
                    <Text style={styles.switchLabel}>{item.label}</Text>
                    <Text style={styles.switchDesc}>{item.desc}</Text>
                  </View>
                </View>
                <Switch
                  value={enabled}
                  onValueChange={v => toggleTypeChannel(item.key, v)}
                  trackColor={{ false: theme.colors.background.tertiary, true: theme.colors.primaryLight }}
                  thumbColor={enabled ? theme.colors.primary : theme.colors.text.tertiary}
                />
              </View>
            </React.Fragment>
          );
        })}
      </View>
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  loadingWrap: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    flex: 1,
  },
  switchTextWrap: {
    flex: 1,
  },
  switchLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  switchDesc: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 4,
  },
  sectionHeader: {
    marginBottom: theme.spacing.sm,
    paddingHorizontal: 4,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  sectionDesc: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
    marginTop: 2,
  },
  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: theme.colors.border.light,
    marginVertical: 12,
  },
});

export default PersonalNotifyScreen;
