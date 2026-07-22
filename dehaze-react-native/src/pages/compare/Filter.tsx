/**
 * 滤镜调节对比模式
 *
 * 对处理后的图片应用滤镜（亮度/对比度/饱和度/色温/锐化/降噪），实时预览调节效果。
 *
 * 实现原理：通过 react-native-svg 的滤镜原语在 GPU 上对图像做真实像素级处理：
 *  - 饱和度       → FeColorMatrix type="saturate"
 *  - 亮度/对比度/色温 → FeColorMatrix type="matrix"（线性变换 + 通道偏移）
 *  - 降噪         → FeGaussianBlur
 *  - 锐化         → FeGaussianBlur + FeComposite(arithmetic) 非锐化掩模
 *
 * 上述原语均为 react-native-svg 已实现的原生滤镜组件，
 * 通过 result/in 链式组合，最终作用于 <Image>。
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  useWindowDimensions,
} from 'react-native';
import {
  Svg,
  Defs,
  Filter,
  FeColorMatrix,
  FeGaussianBlur,
  FeComposite,
  Image as SvgImage,
} from 'react-native-svg';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import CompareModeSwitcher from './components/CompareModeSwitcher';

type Props = NativeStackScreenProps<RootStackParamList, 'Filter'>;

interface FilterParams {
  brightness: number; // -100 ~ +100
  contrast: number; // -100 ~ +100
  saturation: number; // -100 ~ +100
  warmth: number; // -100 ~ +100 (色温)
  sharpen: number; // 0 ~ 100
  denoise: number; // 0 ~ 100
}

const DEFAULT_PARAMS: FilterParams = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  warmth: 0,
  sharpen: 0,
  denoise: 0,
};

const PRESETS: { key: string; name: string; params: FilterParams }[] = [
  { key: 'natural', name: '自然', params: { ...DEFAULT_PARAMS, brightness: 5, contrast: 10, saturation: 5 } },
  { key: 'vivid', name: '鲜艳', params: { ...DEFAULT_PARAMS, contrast: 30, saturation: 40 } },
  { key: 'soft', name: '柔和', params: { ...DEFAULT_PARAMS, contrast: -20, denoise: 15 } },
  { key: 'clear', name: '清晰', params: { ...DEFAULT_PARAMS, contrast: 20, sharpen: 40 } },
  { key: 'vintage', name: '复古', params: { ...DEFAULT_PARAMS, saturation: -20, warmth: 30 } },
];

const PARAM_LIST: { key: keyof FilterParams; label: string; min: number; max: number; step: number }[] = [
  { key: 'brightness', label: '亮度', min: -100, max: 100, step: 1 },
  { key: 'contrast', label: '对比度', min: -100, max: 100, step: 1 },
  { key: 'saturation', label: '饱和度', min: -100, max: 100, step: 1 },
  { key: 'warmth', label: '色温', min: -100, max: 100, step: 1 },
  { key: 'sharpen', label: '锐化', min: 0, max: 100, step: 1 },
  { key: 'denoise', label: '降噪', min: 0, max: 100, step: 1 },
];

const FILTER_ID = 'dehaze-filter';

/**
 * 根据滤镜参数构建 SVG 滤镜原语链。
 * 返回 { primitives, hasFilter }：无任何调节时不应用滤镜。
 */
function buildFilterPrimitives(params: FilterParams) {
  const primitives: React.ReactNode[] = [];
  let currentIn = 'SourceGraphic';
  let seq = 0;
  const nextResult = () => `r${seq++}`;

  // 饱和度：saturate 矩阵（0=灰度，1=原图，>1=增强）
  if (params.saturation !== 0) {
    const saturate = Math.max(0, 1 + params.saturation / 100);
    const out = nextResult();
    primitives.push(
      <FeColorMatrix
        key="saturation"
        in={currentIn}
        type="saturate"
        values={saturate}
        result={out}
      />,
    );
    currentIn = out;
  }

  // 亮度/对比度/色温：单一 matrix 线性变换
  // slope = 亮度斜率 × 对比度斜率；intercept = 对比度截距 × 亮度斜率
  // 色温：R 通道正向偏移、B 通道负向偏移
  if (params.brightness !== 0 || params.contrast !== 0 || params.warmth !== 0) {
    const bSlope = 1 + params.brightness / 100; // 0 ~ 2
    const cSlope = 1 + params.contrast / 100; // 0 ~ 2
    const slope = bSlope * cSlope;
    const intercept = 0.5 * (1 - cSlope) * bSlope;
    const warmthOffset = (params.warmth / 100) * 0.3; // -0.3 ~ 0.3

    const out = nextResult();
    primitives.push(
      <FeColorMatrix
        key="bcm"
        in={currentIn}
        type="matrix"
        values={[
          slope, 0, 0, 0, intercept + warmthOffset,
          0, slope, 0, 0, intercept,
          0, 0, slope, 0, intercept - warmthOffset,
          0, 0, 0, 1, 0,
        ]}
        result={out}
      />,
    );
    currentIn = out;
  }

  // 降噪：高斯模糊
  if (params.denoise > 0) {
    const stdDeviation = (params.denoise / 100) * 2; // 0 ~ 2
    const out = nextResult();
    primitives.push(
      <FeGaussianBlur
        key="denoise"
        in={currentIn}
        stdDeviation={stdDeviation}
        edgeMode="none"
        result={out}
      />,
    );
    currentIn = out;
  }

  // 锐化：非锐化掩模 = 原图 × (1+amount) - 模糊图 × amount
  if (params.sharpen > 0) {
    const amount = (params.sharpen / 100) * 1.5; // 0 ~ 1.5
    const blurOut = nextResult();
    primitives.push(
      <FeGaussianBlur
        key="sharpen-blur"
        in={currentIn}
        stdDeviation={1}
        edgeMode="none"
        result={blurOut}
      />,
    );
    const out = nextResult();
    primitives.push(
      <FeComposite
        key="sharpen-composite"
        in={currentIn}
        in2={blurOut}
        operator="arithmetic"
        k1={0}
        k2={1 + amount}
        k3={-amount}
        k4={0}
        result={out}
      />,
    );
    currentIn = out;
  }

  return { primitives, hasFilter: primitives.length > 0 };
}

const FilterScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl, cleanUrl, algorithmId } = route.params ?? { originalUrl: '', processedUrl: '' };
  const [params, setParams] = useState<FilterParams>({ ...DEFAULT_PARAMS });
  const [showOriginal, setShowOriginal] = useState(false);
  const { width: windowWidth } = useWindowDimensions();

  const svgWidth = windowWidth - theme.spacing.md * 2;
  const svgHeight = 260;

  const { primitives, hasFilter } = buildFilterPrimitives(params);
  // 按住看原图或无任何调节时不应用滤镜
  const applyFilter = hasFilter && !showOriginal;

  const handleStepChange = (key: keyof FilterParams, delta: number) => {
    setParams(prev => {
      const config = PARAM_LIST.find(p => p.key === key);
      if (!config) return prev;
      const current = prev[key];
      const next = Math.max(config.min, Math.min(config.max, current + delta * config.step));
      return { ...prev, [key]: next };
    });
  };

  const handleReset = () => setParams({ ...DEFAULT_PARAMS });

  const handlePreset = (presetParams: FilterParams) => setParams({ ...presetParams });

  const isDefault = JSON.stringify(params) === JSON.stringify(DEFAULT_PARAMS);

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <MainLayout title="滤镜调节" showBack>
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
    <MainLayout title="滤镜调节" showBack>
      <CompareModeSwitcher
        current="Filter"
        navigation={navigation}
        params={{ originalUrl, processedUrl, cleanUrl, algorithmId }}
      />

      {/* 控制栏 */}
      <View style={styles.controlBar}>
        <TouchableOpacity
          style={[styles.controlButton, showOriginal && styles.controlButtonActive]}
          onPressIn={() => setShowOriginal(true)}
          onPressOut={() => setShowOriginal(false)}
        >
          <Icon name="image" size={14} color={showOriginal ? '#fff' : theme.colors.text.secondary} />
          <Text style={[styles.controlText, showOriginal && styles.controlTextActive]}>按住看原图</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.controlButton, isDefault && styles.controlButtonDisabled]}
          onPress={handleReset}
          disabled={isDefault}
        >
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={styles.controlText}>重置</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollView}>
        {/* 预览区 */}
        <View style={styles.previewContainer}>
          <Svg width={svgWidth} height={svgHeight}>
            <Defs>
              <Filter id={FILTER_ID}>{primitives}</Filter>
            </Defs>
            <SvgImage
              href={{ uri: showOriginal ? originalUrl : processedUrl }}
              x="0"
              y="0"
              width="100%"
              height="100%"
              preserveAspectRatio="xMidYMid meet"
              filter={applyFilter ? `url(#${FILTER_ID})` : undefined}
            />
          </Svg>
          <View style={styles.previewLabelRow}>
            <View style={[styles.previewLabel, showOriginal ? styles.previewLabelOriginal : styles.previewLabelResult]}>
              <Text style={styles.previewLabelText}>{showOriginal ? '原图' : '滤镜预览'}</Text>
            </View>
          </View>
        </View>

        {/* 预设方案 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>预设方案</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.presetRow}>
            {PRESETS.map(preset => {
              const isActive = JSON.stringify(params) === JSON.stringify(preset.params);
              return (
                <TouchableOpacity
                  key={preset.key}
                  style={[styles.presetChip, isActive && styles.presetChipActive]}
                  onPress={() => handlePreset(preset.params)}
                >
                  <Text style={[styles.presetText, isActive && styles.presetTextActive]}>
                    {preset.name}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        </View>

        {/* 参数调节 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>参数调节</Text>
          {PARAM_LIST.map(config => {
            const value = params[config.key];
            const fillPercent = ((value - config.min) / (config.max - config.min)) * 100;
            return (
              <View key={config.key} style={styles.paramItem}>
                <View style={styles.paramHeader}>
                  <Text style={styles.paramLabel}>{config.label}</Text>
                  <Text style={styles.paramValue}>{value}</Text>
                </View>
                <View style={styles.sliderRow}>
                  <TouchableOpacity
                    style={styles.stepButton}
                    onPress={() => handleStepChange(config.key, -1)}
                  >
                    <Text style={styles.stepButtonText}>-</Text>
                  </TouchableOpacity>
                  <View style={styles.sliderTrack}>
                    <View style={[styles.sliderFill, { width: `${fillPercent}%` }]} />
                    <View style={[styles.sliderThumb, { left: `${fillPercent}%` }]} />
                  </View>
                  <TouchableOpacity
                    style={styles.stepButton}
                    onPress={() => handleStepChange(config.key, 1)}
                  >
                    <Text style={styles.stepButtonText}>+</Text>
                  </TouchableOpacity>
                </View>
              </View>
            );
          })}
        </View>
      </ScrollView>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
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
  controlButtonActive: {
    backgroundColor: theme.colors.primary,
  },
  controlButtonDisabled: {
    opacity: 0.4,
  },
  controlText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  controlTextActive: {
    color: '#fff',
  },
  previewContainer: {
    margin: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.lg,
    backgroundColor: theme.colors.background.tertiary,
    overflow: 'hidden',
    ...theme.layout.shadows.sm,
  },
  previewLabelRow: {
    position: 'absolute',
    top: theme.spacing.sm,
    left: theme.spacing.sm,
  },
  previewLabel: {
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  previewLabelOriginal: {
    backgroundColor: theme.colors.text.tertiary,
  },
  previewLabelResult: {
    backgroundColor: theme.colors.status.success,
  },
  previewLabelText: {
    fontSize: theme.typography.sizes.tiny,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
  },
  section: {
    backgroundColor: theme.colors.background.primary,
    marginHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.md,
    padding: theme.spacing.lg,
    borderRadius: theme.layout.borderRadius.lg,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.sm,
  },
  presetRow: {
    gap: theme.spacing.sm,
  },
  presetChip: {
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.xs,
    borderRadius: theme.layout.borderRadius.full,
    backgroundColor: theme.colors.background.tertiary,
    borderWidth: 1,
    borderColor: theme.colors.border.light,
  },
  presetChipActive: {
    backgroundColor: `${theme.colors.primary}15`,
    borderColor: theme.colors.primary,
  },
  presetText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  presetTextActive: {
    color: theme.colors.primary,
    fontWeight: theme.typography.weights.medium,
  },
  paramItem: {
    marginBottom: theme.spacing.md,
  },
  paramHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.xs,
  },
  paramLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    color: theme.colors.text.primary,
  },
  paramValue: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
  sliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  stepButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: theme.colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepButtonText: {
    fontSize: 18,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.text.secondary,
  },
  sliderTrack: {
    flex: 1,
    height: 32,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.full,
    position: 'relative',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  sliderFill: {
    position: 'absolute',
    height: '100%',
    backgroundColor: `${theme.colors.primary}40`,
    borderRadius: theme.layout.borderRadius.full,
  },
  sliderThumb: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: theme.colors.primary,
    marginLeft: -10,
    top: '50%',
    marginTop: -10,
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

export default FilterScreen;
