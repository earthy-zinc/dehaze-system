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
  useWindowDimensions,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import CompareEmptyState from '@/components/CompareEmptyState';
import { controlBarStyles, controlButtonStyles } from './styles/compareControls';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<RootStackParamList, 'Overlay'>;

type Direction = 'vertical' | 'horizontal';

const OverlayScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl, cleanUrl, algorithmId } = route.params ?? { originalUrl: '', processedUrl: '' };
  const { height: screenHeight } = useWindowDimensions();
  const [direction, setDirection] = useState<Direction>('vertical');
  const [dividerPos, setDividerPos] = useState(0.5);
  const layoutRef = useRef({ width: 0, height: 0 });
  // 同步 direction 到 ref，避免 PanResponder 闭包捕获陈旧值
  const directionRef = useRef(direction);
  directionRef.current = direction;

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutRef.current = { width, height };
  };

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (evt) => {
        const { width, height } = layoutRef.current;
        const dir = directionRef.current;
        if (dir === 'vertical') {
          const ratio = Math.max(0, Math.min(1, evt.nativeEvent.locationX / width));
          setDividerPos(ratio);
        } else {
          const ratio = Math.max(0, Math.min(1, evt.nativeEvent.locationY / height));
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
        <CompareEmptyState onPress={() => navigation.navigate('ImageInput')} />
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
      <View style={controlBarStyles.bar}>
        <TouchableOpacity style={controlButtonStyles.button} onPress={toggleDirection}>
          <Icon
            name={direction === 'vertical' ? 'arrow-right' : 'arrow-down'}
            size={14}
            color={theme.colors.primary}
          />
          <Text style={controlButtonStyles.text}>
            {direction === 'vertical' ? '垂直分隔' : '水平分隔'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={controlButtonStyles.button}
          onPress={() => setDividerPos(0.5)}
        >
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={controlButtonStyles.text}>居中</Text>
        </TouchableOpacity>
      </View>

      {/* 重叠区域 */}
      <View
        style={[styles.overlayContainer, { minHeight: screenHeight * 0.5 }]}
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
              ? { width: `${dividerPos * 100}%`, height: '100%' }
              : { width: '100%', height: `${dividerPos * 100}%` },
          ]}
        >
          <ImageLoader
            source={{ uri: originalUrl }}
            style={styles.overlayImageFill}
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

const styles = StyleSheet.create({
  overlayContainer: {
    flex: 1,
    margin: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: theme.colors.background.tertiary,
    overflow: 'hidden',
    position: 'relative',
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
  overlayImageFill: {
    width: '100%',
    height: '100%',
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
});

export default OverlayScreen;
