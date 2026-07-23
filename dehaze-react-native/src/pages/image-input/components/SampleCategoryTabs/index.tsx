/**
 * 样例分类标签组件
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { theme } from '@/theme';
import { SampleCategory, CategoryConfig } from '../../types/imageInput';

// 分类配置
const CATEGORIES: CategoryConfig[] = [
  { key: 'all', label: '全部' },
  { key: 'light', label: '轻度雾霾' },
  { key: 'medium', label: '中度雾霾' },
  { key: 'heavy', label: '重度雾霾' },
];

interface SampleCategoryTabsProps {
  currentCategory: SampleCategory;
  onCategoryChange: (category: SampleCategory) => void;
}

const SampleCategoryTabs: React.FC<SampleCategoryTabsProps> = ({
  currentCategory,
  onCategoryChange,
}) => {
  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {CATEGORIES.map(category => {
          const isActive = currentCategory === category.key;

          return (
            <TouchableOpacity
              key={category.key}
              onPress={() => onCategoryChange(category.key)}
              style={[
                styles.tab,
                isActive && styles.tabActive,
              ]}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.tabText,
                  isActive && styles.tabTextActive,
                ]}
              >
                {category.label}
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
    marginBottom: theme.spacing.md,
  },
  scrollContent: {
    paddingHorizontal: theme.spacing.xs,
    gap: theme.spacing.sm,
  },
  tab: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.secondary,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  tabActive: {
    backgroundColor: theme.colors.primary,
    borderColor: theme.colors.primary,
  },
  tabText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.secondary,
  },
  tabTextActive: {
    color: '#fff',
  },
});

export default SampleCategoryTabs;
