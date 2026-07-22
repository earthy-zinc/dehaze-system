/**
 * 对比模式切换器
 *
 * 在 5 种对比模式间快速切换：并排 / 重叠 / 放大镜 / 滤镜 / 指标
 * 当前模式高亮，点击其他模式跳转对应路由（携带相同参数）。
 */
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { theme } from '@/theme';
import Icon from '@/components/Icon';

type CompareMode = 'SideBySide' | 'Overlay' | 'Magnifier' | 'Filter' | 'Metrics';

interface ModeConfig {
  key: CompareMode;
  label: string;
  icon: string;
}

const MODES: ModeConfig[] = [
  { key: 'SideBySide', label: '并排', icon: 'columns' },
  { key: 'Overlay', label: '重叠', icon: 'layer-group' },
  { key: 'Magnifier', label: '放大镜', icon: 'search-plus' },
  { key: 'Filter', label: '滤镜', icon: 'sliders-h' },
  { key: 'Metrics', label: '指标', icon: 'chart-line' },
];

interface CompareModeSwitcherProps {
  current: CompareMode;
  navigation: NativeStackNavigationProp<RootStackParamList>;
  /** 共享参数（原图/处理后URL/GT参考图/算法ID） */
  params: {
    originalUrl: string;
    processedUrl: string;
    cleanUrl?: string;
    algorithmId?: number;
  };
}

const CompareModeSwitcher: React.FC<CompareModeSwitcherProps> = ({
  current,
  navigation,
  params,
}) => {
  const handleSwitch = (mode: CompareMode) => {
    if (mode === current) return;
    navigation.navigate(mode, {
      originalUrl: params.originalUrl,
      processedUrl: params.processedUrl,
      cleanUrl: params.cleanUrl,
      algorithmId: params.algorithmId,
    } as any);
  };

  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {MODES.map(mode => {
          const isActive = mode.key === current;
          return (
            <TouchableOpacity
              key={mode.key}
              style={[styles.tab, isActive && styles.tabActive]}
              onPress={() => handleSwitch(mode.key)}
              activeOpacity={0.7}
            >
              <Icon
                name={mode.icon}
                size={14}
                color={isActive ? theme.colors.primary : theme.colors.text.secondary}
              />
              <Text
                style={[
                  styles.tabText,
                  isActive && styles.tabTextActive,
                ]}
              >
                {mode.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    paddingVertical: theme.spacing.xs,
    paddingHorizontal: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
  scrollContent: {
    gap: theme.spacing.xs,
    alignItems: 'center',
  },
  tab: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
  },
  tabActive: {
    backgroundColor: `${theme.colors.primary}15`,
    borderWidth: 1,
    borderColor: theme.colors.primary,
  },
  tabText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  tabTextActive: {
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
});

export default CompareModeSwitcher;
