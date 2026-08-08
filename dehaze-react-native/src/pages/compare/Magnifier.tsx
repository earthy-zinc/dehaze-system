/**
 * 放大镜对比模式
 *
 * 显示一张图片，在图片上叠加一个可移动的放大镜窗口，
 * 放大镜内同时显示原图与处理后的对比（左右各半）。
 * 提供放大倍数调节（2x / 3x / 5x）。
 */
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  PanResponder,
  useWindowDimensions,
  LayoutChangeEvent,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { DehazeStackParamList } from '@/routes/types';
import { ImmersiveHeader } from '@/layout/components';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import CompareEmptyState from '@/components/CompareEmptyState';
import { controlBarStyles, controlButtonStyles } from './styles/compareControls';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<DehazeStackParamList, 'CompareMagnifier'>;

type DisplayMode = 'original' | 'processed' | 'compare';

const MAGNIFIER_SIZE = 150;

const MagnifierScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl, cleanUrl, algorithmId } = route.params ?? { originalUrl: '', processedUrl: '' };
  const { height: screenHeight } = useWindowDimensions();
  const [zoom, setZoom] = useState<2 | 3 | 5>(2);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('compare');
  const [magnifierPos, setMagnifierPos] = useState({ x: 100, y: 100 });
  const layoutRef = useRef({ width: 0, height: 0 });
  // 保存最新位置与手势起点，避免 PanResponder 闭包捕获陈旧 state
  const posRef = useRef(magnifierPos);
  const startRef = useRef(magnifierPos);
  posRef.current = magnifierPos;

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutRef.current = { width, height };
    // 居中放大镜
    const center = { x: (width - MAGNIFIER_SIZE) / 2, y: (height - MAGNIFIER_SIZE) / 2 };
    setMagnifierPos(center);
  };

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: () => {
        startRef.current = posRef.current;
      },
      onPanResponderMove: (evt, gestureState) => {
        const { width, height } = layoutRef.current;
        const newX = Math.max(0, Math.min(width - MAGNIFIER_SIZE, startRef.current.x + gestureState.dx));
        const newY = Math.max(0, Math.min(height - MAGNIFIER_SIZE, startRef.current.y + gestureState.dy));
        setMagnifierPos({ x: newX, y: newY });
      },
    }),
  ).current;

  const cycleZoom = () => {
    setZoom(z => (z === 2 ? 3 : z === 3 ? 5 : 2));
  };

  const cycleDisplayMode = () => {
    setDisplayMode(m => (m === 'original' ? 'processed' : m === 'processed' ? 'compare' : 'original'));
  };

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <View style={styles.screenContainer}>
        <ImmersiveHeader title="放大镜对比" />
        <CompareEmptyState onPress={() => navigation.goBack()} />
      </View>
    );
  }

  return (
    <View style={styles.screenContainer}>
      <ImmersiveHeader title="放大镜对比" />
      <CompareModeSwitcher
        current="CompareMagnifier"
        navigation={navigation}
        params={{ originalUrl, processedUrl, cleanUrl, algorithmId }}
      />

      {/* 控制栏 */}
      <View style={controlBarStyles.bar}>
        <TouchableOpacity style={controlButtonStyles.button} onPress={cycleZoom}>
          <Icon name="search-plus" size={14} color={theme.colors.primary} />
          <Text style={controlButtonStyles.text}>{zoom}x 放大</Text>
        </TouchableOpacity>
        <TouchableOpacity style={controlButtonStyles.button} onPress={cycleDisplayMode}>
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={controlButtonStyles.text}>
            {displayMode === 'original' ? '仅原图' : displayMode === 'processed' ? '仅去雾后' : '对比模式'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* 图片区域 */}
      <View style={[styles.imageContainer, { minHeight: screenHeight * 0.55 }]} onLayout={handleLayout}>
        <ImageLoader
          source={{ uri: processedUrl }}
          style={styles.baseImage}
          resizeMode="contain"
        />

        {/* 放大镜窗口 */}
        <View
          style={[
            styles.magnifier,
            { left: magnifierPos.x, top: magnifierPos.y },
          ]}
          {...panResponder.panHandlers}
        >
          {displayMode !== 'processed' && (
            <View style={styles.magnifierHalf}>
              <ImageLoader
                source={{ uri: originalUrl }}
                style={{
                  width: MAGNIFIER_SIZE * zoom,
                  height: MAGNIFIER_SIZE * zoom,
                  marginLeft: -magnifierPos.x * zoom,
                  marginTop: -magnifierPos.y * zoom,
                }}
                resizeMode="contain"
              />
            </View>
          )}
          {displayMode !== 'original' && (
            <View
              style={[
                styles.magnifierHalf,
                displayMode === 'compare' && styles.magnifierHalfRight,
              ]}
            >
              <ImageLoader
                source={{ uri: processedUrl }}
                style={{
                  width: MAGNIFIER_SIZE * zoom,
                  height: MAGNIFIER_SIZE * zoom,
                  marginLeft: -magnifierPos.x * zoom,
                  marginTop: -magnifierPos.y * zoom,
                }}
                resizeMode="contain"
              />
            </View>
          )}
          {displayMode === 'compare' && <View style={styles.magnifierDivider} />}
        </View>
      </View>

      {/* 提示 */}
      <View style={styles.tipRow}>
        <Icon name="search" size={12} color={theme.colors.text.tertiary} />
        <Text style={styles.tipText}>拖动放大镜查看局部细节</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  screenContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  imageContainer: {
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
  },
  magnifier: {
    position: 'absolute',
    width: MAGNIFIER_SIZE,
    height: MAGNIFIER_SIZE,
    borderRadius: MAGNIFIER_SIZE / 2,
    overflow: 'hidden',
    backgroundColor: 'transparent',
    borderWidth: 3,
    borderColor: theme.colors.primary,
    ...theme.layout.shadows.lg,
    flexDirection: 'row',
  },
  magnifierHalf: {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  },
  magnifierHalfRight: {
    width: '50%',
    position: 'absolute',
    right: 0,
    top: 0,
  },
  magnifierDivider: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: 2,
    backgroundColor: theme.colors.primary,
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

export default MagnifierScreen;
