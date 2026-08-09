/**
 * 消息管理（管理侧）- 公告/模板/群发
 * 权限：sys:notify:*
 */
import React from 'react';
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Ionicons from 'react-native-vector-icons/Ionicons';

import type { ProfileStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';

type Props = NativeStackScreenProps<ProfileStackParamList, 'SystemMessage'>;

interface EntryItem {
  label: string;
  icon: string;
  route: keyof ProfileStackParamList;
  desc: string;
}

const ENTRIES: EntryItem[] = [
  { label: '公告管理', icon: 'megaphone-outline', route: 'SystemMessageAnnouncement', desc: '创建和管理系统公告' },
  { label: '消息模板', icon: 'document-text-outline', route: 'SystemMessageTemplate', desc: '管理消息模板' },
  { label: '群发消息', icon: 'send-outline', route: 'SystemMessageSend', desc: '向用户群发消息' },
];

const SystemMessageScreen: React.FC<Props> = ({ navigation }) => {
  const handleEntryPress = (route: keyof ProfileStackParamList) => {
    (navigation.navigate as (screen: string) => void)(route as string);
  };

  return (
    <View style={styles.container}>
      <AppHeader title="消息管理" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.container} contentContainerStyle={styles.content}>
        {ENTRIES.map((entry) => (
          <TouchableOpacity
            key={entry.route}
            style={styles.card}
            activeOpacity={0.7}
            onPress={() => handleEntryPress(entry.route)}
          >
            <View style={styles.iconWrap}>
              <Ionicons name={entry.icon} size={28} color={theme.colors.primary} />
            </View>
            <View style={styles.info}>
              <Text style={styles.label}>{entry.label}</Text>
              <Text style={styles.desc}>{entry.desc}</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={theme.colors.text.tertiary} />
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    ...theme.layout.shadows.sm,
  },
  iconWrap: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: theme.colors.primaryLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: theme.spacing.md,
  },
  info: { flex: 1 },
  label: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  desc: { fontSize: theme.typography.sizes.small, color: theme.colors.text.tertiary, marginTop: 2 },
});

export default SystemMessageScreen;
