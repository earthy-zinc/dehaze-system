/**
 * 算法卡片组件
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from 'react-native';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import type { Algorithm } from '@/types/algorithm';

interface AlgorithmCardProps {
  algorithm: Algorithm;
  isSelected?: boolean;
  isFavorite?: boolean;
  matchScore?: number;
  reason?: string;
  onSelect: (algorithm: Algorithm) => void;
  onToggleFavorite: (algorithm: Algorithm) => void;
  onViewDetail: (algorithm: Algorithm) => void;
  onToggleCompare: (algorithm: Algorithm) => void;
}

const AlgorithmCard: React.FC<AlgorithmCardProps> = ({
  algorithm,
  isSelected = false,
  isFavorite = false,
  matchScore,
  reason,
  onSelect,
  onToggleFavorite,
  onViewDetail,
  onToggleCompare,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.97,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  const handlePressOut = () => {
    Animated.spring(scaleAnim, {
      toValue: 1,
      useNativeDriver: true,
      tension: 100,
      friction: 8,
    }).start();
  };

  return (
    <Animated.View
      style={[
        styles.container,
        { transform: [{ scale: scaleAnim }] },
        isSelected && styles.containerSelected,
      ]}
    >
      {/* 头部 */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.iconWrapper}>
            <Icon name="flash" size={18} color={theme.colors.primary} />
          </View>
          <View style={styles.titleContainer}>
            <Text style={styles.name} numberOfLines={1}>
              {algorithm.name}
            </Text>
            {algorithm.type && (
              <View style={styles.typeBadge}>
                <Text style={styles.typeText}>{algorithm.type}</Text>
              </View>
            )}
          </View>
        </View>

        {matchScore !== undefined && (
          <View style={styles.scoreBadge}>
            <Text style={styles.scoreText}>{matchScore}%</Text>
          </View>
        )}
      </View>

      {/* 描述 */}
      {algorithm.description ? (
        <Text style={styles.description} numberOfLines={2}>
          {algorithm.description}
        </Text>
      ) : null}

      {/* 推荐理由 */}
      {reason && (
        <View style={styles.reasonContainer}>
          <Icon name="bulb" size={14} color={theme.colors.status.warning} />
          <Text style={styles.reasonText} numberOfLines={2}>{reason}</Text>
        </View>
      )}

      {/* 元信息 */}
      <View style={styles.metaRow}>
        {algorithm.version && (
          <View style={styles.metaItem}>
            <Icon name="git-branch" size={12} color={theme.colors.text.tertiary} />
            <Text style={styles.metaText}>v{algorithm.version}</Text>
          </View>
        )}
        {algorithm.size && (
          <View style={styles.metaItem}>
            <Icon name="cube" size={12} color={theme.colors.text.tertiary} />
            <Text style={styles.metaText}>{algorithm.size}</Text>
          </View>
        )}
        {algorithm.flops && (
          <View style={styles.metaItem}>
            <Icon name="speedometer" size={12} color={theme.colors.text.tertiary} />
            <Text style={styles.metaText}>{algorithm.flops}</Text>
          </View>
        )}
      </View>

      {/* 操作按钮 */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={[styles.actionButton, styles.primaryAction]}
          onPress={() => onSelect(algorithm)}
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          activeOpacity={0.8}
        >
          <Icon name="rocket" size={16} color="#fff" />
          <Text style={styles.primaryActionText}>使用</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => onViewDetail(algorithm)}
        >
          <Icon name="information-circle-outline" size={18} color={theme.colors.text.secondary} />
          <Text style={styles.actionText}>详情</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => onToggleFavorite(algorithm)}
        >
          <Icon
            name={isFavorite ? 'heart' : 'heart-outline'}
            size={18}
            color={isFavorite ? theme.colors.status.error : theme.colors.text.secondary}
          />
          <Text style={[styles.actionText, isFavorite && styles.favoriteText]}>
            {isFavorite ? '已收藏' : '收藏'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, isSelected && styles.compareSelected]}
          onPress={() => onToggleCompare(algorithm)}
        >
          <Icon
            name={isSelected ? 'checkbox' : 'square-outline'}
            size={18}
            color={isSelected ? theme.colors.primary : theme.colors.text.secondary}
          />
          <Text style={[styles.actionText, isSelected && styles.compareActionSelected]}>
            {isSelected ? '已选' : '对比'}
          </Text>
        </TouchableOpacity>
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.md,
    marginBottom: theme.spacing.sm,
    borderWidth: 2,
    borderColor: theme.colors.border.transparent,
    ...theme.layout.shadows.sm,
  },
  containerSelected: {
    borderColor: theme.colors.primary,
    backgroundColor: `${theme.colors.primary}08`,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.xs,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: theme.spacing.sm,
  },
  iconWrapper: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: `${theme.colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  titleContainer: {
    flex: 1,
    gap: 4,
  },
  name: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  typeBadge: {
    alignSelf: 'flex-start',
    backgroundColor: theme.colors.background.tertiary,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.sm,
  },
  typeText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  scoreBadge: {
    backgroundColor: theme.colors.status.success,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: theme.layout.borderRadius.sm,
  },
  scoreText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  description: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 20,
    marginBottom: theme.spacing.xs,
  },
  reasonContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: `${theme.colors.status.warning}10`,
    borderRadius: theme.layout.borderRadius.sm,
    padding: theme.spacing.xs,
    marginBottom: theme.spacing.xs,
    gap: 4,
  },
  reasonText: {
    flex: 1,
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    lineHeight: 18,
  },
  metaRow: {
    flexDirection: 'row',
    gap: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.tertiary,
  },
  actions: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    marginTop: theme.spacing.xs,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.tertiary,
    gap: 4,
  },
  primaryAction: {
    backgroundColor: theme.colors.primary,
    flex: 1,
  },
  primaryActionText: {
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  actionText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  favoriteText: {
    color: theme.colors.status.error,
  },
  compareSelected: {
    backgroundColor: `${theme.colors.primary}15`,
  },
  compareActionSelected: {
    color: theme.colors.primary,
  },
});

export default AlgorithmCard;
