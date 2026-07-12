/**
 * 算法对比栏组件
 *
 * 底部悬浮栏，显示已选中的对比算法数量，提供对比和清空操作。
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import Icon from '@/components/Icon';
import { theme } from '@/theme';

interface CompareBarProps {
  selectedCount: number;
  maxCount: number;
  onCompare: () => void;
  onClear: () => void;
}

const CompareBar: React.FC<CompareBarProps> = ({
  selectedCount,
  maxCount,
  onCompare,
  onClear,
}) => {
  if (selectedCount === 0) return null;

  const canCompare = selectedCount >= 2;

  return (
    <View style={styles.container}>
      <View style={styles.left}>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{selectedCount}/{maxCount}</Text>
        </View>
        <Text style={styles.text}>
          {canCompare ? '可对比选中的算法' : '再选择 1 个即可对比'}
        </Text>
      </View>

      <View style={styles.right}>
        <TouchableOpacity
          style={styles.clearButton}
          onPress={onClear}
        >
          <Icon name="close-circle" size={18} color={theme.colors.text.secondary} />
          <Text style={styles.clearText}>清空</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.compareButton, !canCompare && styles.compareButtonDisabled]}
          onPress={onCompare}
          disabled={!canCompare}
          activeOpacity={0.8}
        >
          <Icon name="swap-horizontal" size={18} color="#fff" />
          <Text style={styles.compareText}>对比</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: theme.colors.background.primary,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    borderTopWidth: 1,
    borderTopColor: theme.colors.border.light,
    ...theme.layout.shadows.lg,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    flex: 1,
  },
  badge: {
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  badgeText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  text: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  right: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.md,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  clearText: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  compareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.primary,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md,
    gap: 4,
  },
  compareButtonDisabled: {
    backgroundColor: theme.colors.text.muted,
  },
  compareText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
});

export default CompareBar;
