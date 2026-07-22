/**
 * 重叠对比模式
 *
 * 两张图叠加显示，通过拖动分隔线对比左右（或上下）区域。
 * 提供方向切换（垂直/水平分隔线）。
 */
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  PanResponder,
  LayoutChangeEvent,
  Dimensions,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<RootStackParamList, 'Overlay'>;

type Direction = 'vertical' | 'horizontal';

const OverlayScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl, cleanUrl, algorithmId } = route.params ?? { originalUrl: '', processedUrl: '' };
  const [direction, setDirection] = useState<Direction>('vertical');
  const [dividerPos, setDividerPos] = useState(0.5);
  const layoutRef = useRef({ width: 0, height: 0 });

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutRef.current = { width, height };
  };

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (evt, gestureState) => {
        const { width, height } = layoutRef.current;
        if (direction === 'vertical') {
          const ratio = Math.max(0, Math.min(1, (gestureState.moveX || evt.nativeEvent.pageX) / width));
          setDividerPos(ratio);
        } else {
          const ratio = Math.max(0, Math.min(1, (gestureState.moveY || evt.nativeEvent.pageY) / height));
          setDividerPos(ratio);
        }
      },
    }),
  ).current;

  const toggleDirection = () => {
    setDirection(d => (d === 'vertical' ? 'horizontal' : 'vertical'));
    setDividerPos(0.5);
  };

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <MainLayout title="重叠对比" showBack>
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
    <MainLayout title="重叠对比" showBack>
      <CompareModeSwitcher
        current="Overlay"
        navigation={navigation}
        params={{ originalUrl, processedUrl, cleanUrl, algorithmId }}
      />

      {/* 控制栏 */}
      <View style={styles.controlBar}>
        <TouchableOpacity style={styles.controlButton} onPress={toggleDirection}>
          <Icon
            name={direction === 'vertical' ? 'arrow-right' : 'arrow-down'}
            size={14}
            color={theme.colors.primary}
          />
          <Text style={styles.controlText}>
            {direction === 'vertical' ? '垂直分隔' : '水平分隔'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.controlButton}
          onPress={() => setDividerPos(0.5)}
        >
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={styles.controlText}>居中</Text>
        </TouchableOpacity>
      </View>

      {/* 重叠区域 */}
      <View
        style={styles.overlayContainer}
        onLayout={handleLayout}
        {...panResponder.panHandlers}
      >
        {/* 底层：处理后图片 */}
        <ImageLoader
          source={{ uri: processedUrl }}
          style={styles.baseImage}
          resizeMode="contain"
        />

        {/* 上层：原图，通过 clip 显示左/上半部分 */}
        <View
          style={[
            styles.overlayImage,
            direction === 'vertical'
              ? { width: `${dividerPos * 100}%` }
              : { height: `${dividerPos * 100}%` },
          ]}
        >
          <ImageLoader
            source={{ uri: originalUrl }}
            style={
              direction === 'vertical'
                ? { width: layoutRef.current.width || '100%' as any, height: '100%' as any }
                : { width: '100%' as any, height: layoutRef.current.height || '100%' as any }
            }
            resizeMode="contain"
          />
        </View>

        {/* 分隔线 */}
        <View
          style={[
            styles.divider,
            direction === 'vertical'
              ? { left: `${dividerPos * 100}%` }
              : { top: `${dividerPos * 100}%` },
          ]}
        >
          <View
            style={[
              styles.dividerLine,
              direction === 'vertical'
                ? styles.dividerLineVertical
                : styles.dividerLineHorizontal,
            ]}
          />
          <View style={styles.dividerHandle}>
            <Icon name="search" size={14} color={theme.colors.primary} />
          </View>
        </View>

        {/* 标签 */}
        <View style={[styles.label, styles.labelOriginal]}>
          <Text style={styles.labelText}>原图</Text>
        </View>
        <View style={[styles.label, styles.labelResult]}>
          <Text style={styles.labelText}>去雾后</Text>
        </View>
      </View>

      {/* 提示 */}
      <View style={styles.tipRow}>
        <Icon name="search" size={12} color={theme.colors.text.tertiary} />
        <Text style={styles.tipText}>拖动分隔线对比左右/上下区域</Text>
      </View>
    </MainLayout>
  );
};

const { height: screenHeight } = Dimensions.get('window');

const styles = StyleSheet.create({
  controlBar: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    backgroundColor: theme.colors.background.primary,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.border.light,
  },
  controlButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
  },
  controlText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  overlayContainer: {
    flex: 1,
    margin: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: theme.colors.background.tertiary,
    overflow: 'hidden',
    position: 'relative',
    minHeight: screenHeight * 0.5,
  },
  baseImage: {
    width: '100%',
    height: '100%',
    position: 'absolute',
  },
  overlayImage: {
    position: 'absolute',
    overflow: 'hidden',
    top: 0,
    left: 0,
    backgroundColor: 'transparent',
  },
  divider: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  dividerLine: {
    backgroundColor: theme.colors.primary,
  },
  dividerLineVertical: {
    width: 2,
    height: '100%',
  },
  dividerLineHorizontal: {
    height: 2,
    width: '100%',
  },
  dividerHandle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: theme.colors.background.primary,
    justifyContent: 'center',
    alignItems: 'center',
    ...theme.layout.shadows.md,
  },
  label: {
    position: 'absolute',
    top: theme.spacing.sm,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  labelOriginal: {
    left: theme.spacing.sm,
    backgroundColor: theme.colors.text.tertiary,
  },
  labelResult: {
    right: theme.spacing.sm,
    backgroundColor: theme.colors.status.success,
  },
  labelText: {
    fontSize: theme.typography.sizes.tiny,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  tipRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    paddingVertical: theme.spacing.md,
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

export default OverlayScreen;
