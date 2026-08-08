/**
 * 关于我们 (L2)
 *
 * Logo + 版本号 + 简介 + 隐私政策/用户协议
 */
import React, { useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Linking,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';

import { theme } from '@/theme';
import { AppHeader } from '@/layout';

const PersonalAboutScreen: React.FC = () => {
  const navigation = useNavigation();
  const handleOpenPrivacy = useCallback(() => {
    Linking.openURL('https://dehaze.example.com/privacy').catch(() => {});
  }, []);

  const handleOpenTerms = useCallback(() => {
    Linking.openURL('https://dehaze.example.com/terms').catch(() => {});
  }, []);

  return (
    <View style={styles.container}>
      <AppHeader title="关于我们" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.content}>
      {/* Logo 区域 */}
      <View style={styles.logoSection}>
        <View style={styles.logoWrap}>
          <Ionicons name="water-outline" size={48} color={theme.colors.primary} />
        </View>
        <Text style={styles.appName}>Dehaze 图像去雾</Text>
        <Text style={styles.version}>版本 1.0.0</Text>
      </View>

      {/* 简介 */}
      <View style={styles.card}>
        <Text style={styles.descTitle}>产品简介</Text>
        <Text style={styles.descText}>
          Dehaze 是一款基于深度学习技术的智能图像去雾系统，支持多种先进算法，
          可高效去除图像中的雾霾、提高图像清晰度与对比度。
          系统提供上传处理、效果对比、批量处理、数据集管理等功能，
          满足科研与工业场景下的图像增强需求。
        </Text>
      </View>

      {/* 技术栈 */}
      <View style={styles.card}>
        <Text style={styles.descTitle}>技术栈</Text>
        <Text style={styles.descText}>
          React Native 0.81 · React 19 · TypeScript{'\n'}
          @react-navigation v7 · zustand{'\n'}
          后端：Java Spring Boot · Python · Go
        </Text>
      </View>

      {/* 链接 */}
      <View style={styles.card}>
        <TouchableOpacity style={styles.linkRow} onPress={handleOpenPrivacy} activeOpacity={0.6}>
          <Ionicons name="shield-checkmark-outline" size={20} color={theme.colors.text.secondary} />
          <Text style={styles.linkText}>隐私政策</Text>
          <Ionicons name="chevron-forward" size={16} color={theme.colors.text.tertiary} />
        </TouchableOpacity>
        <View style={styles.linkDivider} />
        <TouchableOpacity style={styles.linkRow} onPress={handleOpenTerms} activeOpacity={0.6}>
          <Ionicons name="document-text-outline" size={20} color={theme.colors.text.secondary} />
          <Text style={styles.linkText}>用户协议</Text>
          <Ionicons name="chevron-forward" size={16} color={theme.colors.text.tertiary} />
        </TouchableOpacity>
      </View>

      <Text style={styles.copyright}>© 2024 Dehaze Team. All rights reserved.</Text>
    </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  content: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl, alignItems: 'center' },
  logoSection: {
    alignItems: 'center',
    paddingVertical: theme.spacing.xl,
  },
  logoWrap: {
    width: 96,
    height: 96,
    borderRadius: 24,
    backgroundColor: theme.colors.primaryLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  appName: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
  },
  version: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.tertiary,
    marginTop: 4,
  },
  card: {
    width: '100%',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  descTitle: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  descText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 22,
  },
  linkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    gap: theme.spacing.sm,
  },
  linkText: {
    flex: 1,
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
    fontWeight: theme.typography.weights.medium,
  },
  linkDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: theme.colors.border.light,
    marginVertical: 12,
  },
  copyright: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
    marginTop: theme.spacing.lg,
  },
});

export default PersonalAboutScreen;
