/**
 * 样例图片卡片组件
 */

import React, { useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
} from 'react-native';
import ImageLoader from '@/components/ImageLoader';
import Icon from '@/components/Icon';
import { theme } from '@/theme';
import { SampleImage, DifficultyLevel } from '../../types/imageInput';

interface SampleImageCardProps {
  sample: SampleImage;
  onPress: (sample: SampleImage) => void;
}

// 难度配置
const DIFFICULTY_CONFIG: Record<DifficultyLevel, { label: string; color: string; bgColor: string }> = {
  easy: {
    label: '简单',
    color: theme.colors.status.success,
    bgColor: `${theme.colors.status.success}20`,
  },
  medium: {
    label: '中等',
    color: theme.colors.status.warning,
    bgColor: `${theme.colors.status.warning}20`,
  },
  hard: {
    label: '困难',
    color: theme.colors.status.error,
    bgColor: `${theme.colors.status.error}20`,
  },
};

const SampleImageCard: React.FC<SampleImageCardProps> = ({
  sample,
  onPress,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.95,
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

  const difficultyConfig = DIFFICULTY_CONFIG[sample.difficulty];

  return (
    <TouchableOpacity
      onPress={() => onPress(sample)}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      style={styles.container}
    >
      <Animated.View
        style={[
          styles.card,
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        {/* 图片 */}
        <View style={styles.imageContainer}>
          <ImageLoader
            source={{ uri: sample.thumbUrl || sample.url }}
            style={styles.image}
            resizeMode="cover"
          />

          {/* 场景类型标签 */}
          {sample.sceneType && (
            <View style={styles.sceneTag}>
              <Text style={styles.sceneTagText}>{sample.sceneType}</Text>
            </View>
          )}
        </View>

        {/* 信息区域 */}
        <View style={styles.infoContainer}>
          <Text style={styles.name} numberOfLines={1}>
            {sample.name}
          </Text>

          <View style={styles.footer}>
            {/* 难度标签 */}
            <View
              style={[
                styles.difficultyBadge,
                { backgroundColor: difficultyConfig.bgColor },
              ]}
            >
              <Text
                style={[
                  styles.difficultyText,
                  { color: difficultyConfig.color },
                ]}
              >
                {difficultyConfig.label}
              </Text>
            </View>

            {/* 箭头 */}
            <Icon
              name="arrow-forward"
              size={16}
              color={theme.colors.primary}
            />
          </View>
        </View>
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 6,
  },
  card: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    overflow: 'hidden',
    ...theme.layout.shadows.md,
  },
  imageContainer: {
    height: 120,
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  sceneTag: {
    position: 'absolute',
    top: theme.spacing.xs,
    left: theme.spacing.xs,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.sm,
  },
  sceneTagText: {
    fontSize: theme.typography.sizes.small,
    color: '#fff',
  },
  infoContainer: {
    padding: theme.spacing.sm,
  },
  name: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.xs,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  difficultyBadge: {
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  difficultyText: {
    fontSize: theme.typography.sizes.small,
    fontWeight: theme.typography.weights.medium,
  },
});

export default SampleImageCard;
