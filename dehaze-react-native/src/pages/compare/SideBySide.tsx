/**
 * 并排对比模式
 *
 * 移动端默认上下排列，平板/桌面左右排列。
 * 提供双击放大还原、标签切换原图/处理后。
 */
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions,
  ViewStyle,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import { useResponsive } from '@/hooks/useResponsive';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<RootStackParamList, 'SideBySide'>;

type DisplayMode = 'both' | 'original' | 'processed';

const SideBySideScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl } = route.params ?? { originalUrl: '', processedUrl: '' };
  const { isPortrait, isTablet, isDesktop, containerPadding } = useResponsive();
  const [displayMode, setDisplayMode] = useState<DisplayMode>('both');
  const [zoomed, setZoomed] = useState(false);
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const toggleZoom = () => {
    const next = !zoomed;
    setZoomed(next);
    Animated.spring(scaleAnim, {
      toValue: next ? 1.5 : 1,
      useNativeDriver: true,
      tension: 80,
      friction: 8,
    }).start();
  };

  const isHorizontal = !isPortrait || isTablet || isDesktop;
  const containerStyle: ViewStyle = isHorizontal
    ? { flexDirection: 'row' as const }
    : { flexDirection: 'column' as const };

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <MainLayout title="并排对比" showBack>
        <View style={styles.emptyContainer}>
          <Icon name="image" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.emptyTitle}>请先完成去雾处理</Text>
          <Text style={styles.emptyDesc}>对比功能需要先处理图片</Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={() => navigation.navigate('ImageInput')}
          >
            <Text style={styles.emptyButtonText}>去选择图片</Text>
          </TouchableOpacity>
        </View>
      </MainLayout>
    );
  }

  return (
    <MainLayout title="并排对比" showBack>
      <CompareModeSwitcher
        current="SideBySide"
        navigation={navigation}
        params={{ originalUrl, processedUrl }}
      />

      {/* 模式切换标签 */}
      <View style={styles.tabBar}>
        {(['both', 'original', 'processed'] as DisplayMode[]).map(m => (
          <TouchableOpacity
            key={m}
            style={[styles.tabItem, displayMode === m && styles.tabItemActive]}
            onPress={() => setDisplayMode(m)}
          >
            <Text
              style={[
                styles.tabText,
                displayMode === m && styles.tabTextActive,
              ]}
            >
              {m === 'both' ? '全部' : m === 'original' ? '原图' : '去雾后'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={{ padding: containerPadding }}
      >
        <Animated.View style={[styles.imageContainer, containerStyle, { transform: [{ scale: scaleAnim }] }]}>
          {(displayMode === 'both' || displayMode === 'original') && (
            <View style={styles.imageItem}>
              <View style={styles.imageLabelRow}>
                <View style={[styles.imageBadge, styles.imageBadgeOriginal]}>
                  <Text style={styles.imageBadgeText}>原图</Text>
                </View>
              </View>
              <TouchableOpacity onPress={toggleZoom} activeOpacity={1}>
                <ImageLoader
                  source={{ uri: originalUrl }}
                  style={isHorizontal ? styles.imageHorizontal : styles.imageVertical}
                  resizeMode="contain"
                />
              </TouchableOpacity>
            </View>
          )}

          {(displayMode === 'both' || displayMode === 'processed') && (
            <View style={styles.imageItem}>
              <View style={styles.imageLabelRow}>
                <View style={[styles.imageBadge, styles.imageBadgeResult]}>
                  <Text style={styles.imageBadgeText}>去雾后</Text>
                </View>
              </View>
              <TouchableOpacity onPress={toggleZoom} activeOpacity={1}>
                <ImageLoader
                  source={{ uri: processedUrl }}
                  style={isHorizontal ? styles.imageHorizontal : styles.imageVertical}
                  resizeMode="contain"
                />
              </TouchableOpacity>
            </View>
          )}
        </Animated.View>

        {/* 提示 */}
        <View style={styles.tipRow}>
          <Icon name="search" size={12} color={theme.colors.text.tertiary} />
          <Text style={styles.tipText}>双击图片可放大查看</Text>
        </View>
      </ScrollView>
    </MainLayout>
  );
};

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: theme.colors.background.primary,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    gap: theme.spacing.xs,
  },
  tabItem: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md,
    alignItems: 'center',
    backgroundColor: theme.colors.background.tertiary,
  },
  tabItemActive: {
    backgroundColor: theme.colors.primary,
  },
  tabText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  tabTextActive: {
    color: '#fff',
    fontWeight: theme.typography.weights.semibold,
  },
  imageContainer: {
    gap: theme.spacing.md,
    alignItems: 'stretch',
  },
  imageItem: {
    flex: 1,
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    overflow: 'hidden',
    ...theme.layout.shadows.sm,
  },
  imageLabelRow: {
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: theme.spacing.xs,
    backgroundColor: theme.colors.background.tertiary,
  },
  imageBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  imageBadgeOriginal: {
    backgroundColor: theme.colors.text.tertiary,
  },
  imageBadgeResult: {
    backgroundColor: theme.colors.status.success,
  },
  imageBadgeText: {
    fontSize: theme.typography.sizes.tiny,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  imageVertical: {
    width: '100%',
    height: screenHeight * 0.35,
  },
  imageHorizontal: {
    width: (screenWidth - 60) / 2,
    height: screenHeight * 0.5,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.lg,
  },
  tipText: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: theme.spacing.xl,
  },
  emptyTitle: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginTop: theme.spacing.md,
    marginBottom: theme.spacing.xs,
  },
  emptyDesc: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.lg,
    textAlign: 'center',
  },
  emptyButton: {
    paddingHorizontal: theme.spacing.xl,
    paddingVertical: theme.spacing.md,
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.md,
  },
  emptyButtonText: {
    color: '#fff',
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
  },
});

export default SideBySideScreen;
