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
import CompareEmptyState from '@/components/CompareEmptyState';
import SliderControl from '@/components/SliderControl';
import { controlBarStyles, controlButtonStyles } from './styles/compareControls';
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

/** 逐字段比较两份滤镜参数（避免 JSON.stringify 的字段顺序敏感问题） */
function isFilterParamsEqual(a: FilterParams, b: FilterParams): boolean {
  return (
    a.brightness === b.brightness &&
    a.contrast === b.contrast &&
    a.saturation === b.saturation &&
    a.warmth === b.warmth &&
    a.sharpen === b.sharpen &&
    a.denoise === b.denoise
  );
}

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

  const handleValueChange = (key: keyof FilterParams, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const handleReset = () => setParams({ ...DEFAULT_PARAMS });

  const handlePreset = (presetParams: FilterParams) => setParams({ ...presetParams });

  const isDefault = isFilterParamsEqual(params, DEFAULT_PARAMS);

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <MainLayout title="滤镜调节" showBack>
        <CompareEmptyState onPress={() => navigation.navigate('ImageInput')} />
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
      <View style={controlBarStyles.bar}>
        <TouchableOpacity
          style={[controlButtonStyles.button, showOriginal && controlButtonStyles.buttonActive]}
          onPressIn={() => setShowOriginal(true)}
          onPressOut={() => setShowOriginal(false)}
        >
          <Icon name="image" size={14} color={showOriginal ? '#fff' : theme.colors.text.secondary} />
          <Text style={[controlButtonStyles.text, showOriginal && controlButtonStyles.textActive]}>按住看原图</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[controlButtonStyles.button, isDefault && controlButtonStyles.buttonDisabled]}
          onPress={handleReset}
          disabled={isDefault}
        >
          <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
          <Text style={controlButtonStyles.text}>重置</Text>
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
              const isActive = isFilterParamsEqual(params, preset.params);
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
            return (
              <View key={config.key} style={styles.paramItem}>
                <View style={styles.paramHeader}>
                  <Text style={styles.paramLabel}>{config.label}</Text>
                  <Text style={styles.paramValue}>{value}</Text>
                </View>
                <SliderControl
                  value={value}
                  min={config.min}
                  max={config.max}
                  step={config.step}
                  onChange={v => handleValueChange(config.key, v)}
                />
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
    fontSize: theme.typography.sizes.medium,
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
    fontSize: theme.typography.sizes.medium,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
});

export default FilterScreen;
