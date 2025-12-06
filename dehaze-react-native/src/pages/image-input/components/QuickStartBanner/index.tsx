/**
 * 快速体验横幅组件
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

interface QuickStartBannerProps {
  onQuickStart: () => void;
  loading?: boolean;
}

const QuickStartBanner: React.FC<QuickStartBannerProps> = ({
  onQuickStart,
  loading = false,
}) => {
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const handlePressIn = () => {
    Animated.spring(scaleAnim, {
      toValue: 0.98,
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
    <TouchableOpacity
      onPress={onQuickStart}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      activeOpacity={1}
      disabled={loading}
    >
      <Animated.View
        style={[
          styles.container,
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        <View style={styles.content}>
          <View style={styles.iconContainer}>
            <Icon name="flash" size={24} color="#fff" />
          </View>

          <View style={styles.textContainer}>
            <Text style={styles.title}>快速体验</Text>
            <Text style={styles.description}>
              使用样例图片快速体验去雾效果
            </Text>
          </View>

          <View style={styles.buttonContainer}>
            <View style={styles.button}>
              <Text style={styles.buttonText}>
                {loading ? '加载中...' : '立即体验'}
              </Text>
              {!loading && (
                <Icon name="arrow-forward" size={16} color={theme.colors.primary} />
              )}
            </View>
          </View>
        </View>

        {/* 装饰元素 */}
        <View style={styles.decorCircle1} />
        <View style={styles.decorCircle2} />
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.primary,
    borderRadius: theme.layout.borderRadius.xl,
    padding: theme.spacing.xl,
    overflow: 'hidden',
    ...theme.layout.shadows.lg,
    shadowColor: theme.colors.primary,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 1,
  },
  iconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: theme.spacing.md,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    fontSize: theme.typography.sizes.h5,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
    marginBottom: 4,
  },
  description: {
    fontSize: theme.typography.sizes.caption,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  buttonContainer: {
    marginLeft: theme.spacing.md,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.md,
    gap: 4,
  },
  buttonText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.primary,
  },
  // 装饰圆形
  decorCircle1: {
    position: 'absolute',
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    top: -30,
    right: -20,
  },
  decorCircle2: {
    position: 'absolute',
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    bottom: -20,
    right: 60,
  },
});

export default QuickStartBanner;
