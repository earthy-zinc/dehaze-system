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
  Dimensions,
  LayoutChangeEvent,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<RootStackParamList, 'Magnifier'>;

type DisplayMode = 'original' | 'processed' | 'compare';

const MAGNIFIER_SIZE = 150;

const MagnifierScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl } = route.params ?? { originalUrl: '', processedUrl: '' };
  const [zoom, setZoom] = useState<2 | 3 | 5>(2);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('compare');
  const [magnifierPos, setMagnifierPos] = useState({ x: 100, y: 100 });
  const layoutRef = useRef({ width: 0, height: 0 });

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    layoutRef.current = { width, height };
    // 居中放大镜
    setMagnifierPos({ x: (width - MAGNIFIER_SIZE) / 2, y: (height - MAGNIFIER_SIZE) / 2 });
  };

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: () => true,
      onPanResponderMove: (evt, gestureState) => {
        const { width, height } = layoutRef.current;
        const newX = Math.max(0, Math.min(width - MAGNIFIER_SIZE, magnifierPos.x + gestureState.dx));
        const newY = Math.max(0, Math.min(height - MAGNIFIER_SIZE, magnifierPos.y + gestureState.dy));
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

  return (
    <MainLayout title="放大镜对比" showBack>
      <CompareModeSwitcher
        current="Magnifier"
        navigation={navigation}
        params={{ originalUrl, processedUrl }}
      />

      {/* 控制栏 */}
      <View style={styles.controlBar}>
        <TouchableOpacity style={styles.controlButton} onPress={cycleZoom}>
          <Icon name="search-plus" size={14} color={theme.colors.primary} />
          <Text style={styles.controlText}>{zoom}x 放大</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.controlButton} onPress={cycleDisplayMode}>
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={styles.controlText}>
            {displayMode === 'original' ? '仅原图' : displayMode === 'processed' ? '仅去雾后' : '对比模式'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* 图片区域 */}
      <View style={styles.imageContainer} onLayout={handleLayout}>
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
    </MainLayout>
  );
};

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

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
  imageContainer: {
    flex: 1,
    margin: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: theme.colors.background.tertiary,
    overflow: 'hidden',
    position: 'relative',
    minHeight: screenHeight * 0.55,
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
