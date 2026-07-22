/**
 * 指标评估对比模式
 *
 * 调用 SDK ModelAPI.evaluate 计算 PSNR/SSIM/LPIPS 等指标，
 * 提供柱状图与雷达图可视化。
 *
 * 评估调用条件：需 algorithmId，否则仅展示预先传入的 metrics 或提示无法评估。
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import { ModelAPI } from 'dehaze-sdk-js';
import type { EvaluationResultVO } from 'dehaze-sdk-js';
import CompareModeSwitcher from './components/CompareModeSwitcher';
import type { EvaluationMetrics } from '@/types/evaluation';

type Props = NativeStackScreenProps<RootStackParamList, 'Metrics'>;

interface MetricItem {
  key: string;
  label: string;
  value?: number;
  unit?: string;
  idealRange?: string;
  better?: 'higher' | 'lower';
  description?: string;
}

const MetricsScreen: React.FC<Props> = ({ route, navigation }) => {
  const { originalUrl, processedUrl, metrics: presetMetrics, algorithmId } = route.params ?? {
    originalUrl: '',
    processedUrl: '',
  };
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvaluationResultVO | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleEvaluate = useCallback(async () => {
    if (!algorithmId) {
      setError('缺少算法 ID，无法调用评估接口');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await ModelAPI.evaluate({
        algorithmId,
        predUrl: processedUrl,
        gtUrl: originalUrl,
      });
      setResult(res);
    } catch (err) {
      const msg = err instanceof Error ? err.message : '评估失败';
      setError(msg);
      Alert.alert('评估失败', msg);
    } finally {
      setLoading(false);
    }
  }, [algorithmId, originalUrl, processedUrl]);

  // 自动评估一次（如果有 algorithmId 且没有预传入 metrics）
  useEffect(() => {
    if (!presetMetrics && algorithmId) {
      handleEvaluate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 合并指标（预传入 metrics 优先，否则使用评估结果）
  const metricsList: MetricItem[] = buildMetricsList(presetMetrics, result);

  // 取最大值用于柱状图归一化
  const maxValue = Math.max(...metricsList.map(m => m.value ?? 0), 1);

  // 缺少必要参数时显示空状态（例如从底部 Tab 直接进入）
  if (!originalUrl || !processedUrl) {
    return (
      <MainLayout title="指标评估" showBack>
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
    <MainLayout title="指标评估" showBack>
      <CompareModeSwitcher
        current="Metrics"
        navigation={navigation}
        params={{ originalUrl, processedUrl, algorithmId }}
      />

      <ScrollView style={styles.scrollView}>
        {/* 评估状态 */}
        <View style={styles.statusCard}>
          <View style={styles.statusHeader}>
            <Icon name="chart-line" size={18} color={theme.colors.primary} />
            <Text style={styles.statusTitle}>效果评估</Text>
          </View>
          <Text style={styles.statusDesc}>
            {algorithmId
              ? '点击下方按钮重新评估，获取最新量化指标'
              : '当前未携带算法信息，仅展示预传入指标'}
          </Text>

          {loading && (
            <View style={styles.loadingRow}>
              <ActivityIndicator size="small" color={theme.colors.primary} />
              <Text style={styles.loadingText}>评估中...</Text>
            </View>
          )}

          {error && (
            <View style={styles.errorRow}>
              <Icon name="times" size={14} color={theme.colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {algorithmId && (
            <TouchableOpacity
              style={[styles.evaluateButton, loading && styles.evaluateButtonDisabled]}
              onPress={handleEvaluate}
              disabled={loading}
            >
              <Icon name="refresh" size={14} color="#fff" />
              <Text style={styles.evaluateButtonText}>
                {loading ? '评估中...' : '重新评估'}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {/* 指标列表（柱状图） */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>图像质量指标</Text>
          {metricsList.map(item => (
            <View key={item.key} style={styles.metricItem}>
              <View style={styles.metricHeader}>
                <View style={styles.metricLabelRow}>
                  <Text style={styles.metricLabel}>{item.label}</Text>
                  {item.idealRange && (
                    <Text style={styles.metricIdeal}>理想：{item.idealRange}</Text>
                  )}
                </View>
                <Text style={styles.metricValue}>
                  {item.value !== undefined ? formatMetricValue(item.value, item.unit) : '—'}
                </Text>
              </View>
              {item.value !== undefined && (
                <View style={styles.barContainer}>
                  <View
                    style={[
                      styles.barFill,
                      {
                        width: `${(item.value / maxValue) * 100}%`,
                        backgroundColor: getBarColor(item.key, item.value),
                      },
                    ]}
                  />
                </View>
              )}
              {item.description && (
                <Text style={styles.metricDesc}>{item.description}</Text>
              )}
            </View>
          ))}
        </View>

        {/* 原图/结果预览 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>对比预览</Text>
          <View style={styles.previewRow}>
            <View style={styles.previewItem}>
              <View style={styles.previewBadgeRow}>
                <View style={[styles.previewBadge, styles.previewBadgeOriginal]}>
                  <Text style={styles.previewBadgeText}>原图</Text>
                </View>
              </View>
              <ImageLoaderMini url={originalUrl} />
            </View>
            <View style={styles.previewItem}>
              <View style={styles.previewBadgeRow}>
                <View style={[styles.previewBadge, styles.previewBadgeResult]}>
                  <Text style={styles.previewBadgeText}>去雾后</Text>
                </View>
              </View>
              <ImageLoaderMini url={processedUrl} />
            </View>
          </View>
        </View>
      </ScrollView>
    </MainLayout>
  );
};

/** 简化的图片预览组件 */
const ImageLoaderMini: React.FC<{ url: string }> = ({ url }) => {
  // 延迟导入避免循环依赖
  const ImageLoader = require('@/components/ImageLoader').default;
  return (
    <ImageLoader
      source={{ uri: url }}
      style={metricsStyles.previewImage}
      resizeMode="contain"
    />
  );
};

/** 构造指标列表（合并预传入与评估结果） */
function buildMetricsList(
  preset: EvaluationMetrics | undefined,
  result: EvaluationResultVO | null,
): MetricItem[] {
  const metricsSource: Record<string, number | undefined> = {};
  // 评估结果优先
  if (result?.metrics) {
    Object.assign(metricsSource, result.metrics);
  }
  // 预传入指标补充
  if (preset) {
    Object.entries(preset).forEach(([k, v]) => {
      if (v !== undefined && metricsSource[k] === undefined) {
        metricsSource[k] = v;
      }
    });
  }

  return [
    {
      key: 'psnr',
      label: 'PSNR 峰值信噪比',
      value: metricsSource.psnr,
      unit: 'dB',
      idealRange: '>25',
      better: 'higher',
      description: '图像质量评价，越高越好',
    },
    {
      key: 'ssim',
      label: 'SSIM 结构相似性',
      value: metricsSource.ssim,
      idealRange: '>0.85',
      better: 'higher',
      description: '结构相似程度，越接近 1 越好',
    },
    {
      key: 'mse',
      label: 'MSE 均方误差',
      value: metricsSource.mse,
      idealRange: '<100',
      better: 'lower',
      description: '图像差异程度，越小越好',
    },
    {
      key: 'entropy',
      label: 'Entropy 信息熵',
      value: metricsSource.entropy,
      idealRange: '7-8',
      description: '图像信息量，适中为好',
    },
    {
      key: 'lpips',
      label: 'LPIPS 感知差异',
      value: metricsSource.lpips,
      idealRange: '<0.3',
      better: 'lower',
      description: '感知差异，越小越好',
    },
    {
      key: 'niqe',
      label: 'NIQE 自然图像质量',
      value: metricsSource.niqe,
      idealRange: '<5',
      better: 'lower',
      description: '自然图像质量评估，越小越好',
    },
    {
      key: 'contrastGain',
      label: '对比度提升',
      value: metricsSource.contrastGain,
      unit: '%',
      idealRange: '20%-50%',
      better: 'higher',
      description: '对比度增加程度',
    },
    {
      key: 'saturationGain',
      label: '饱和度提升',
      value: metricsSource.saturationGain,
      unit: '%',
      idealRange: '10%-30%',
      better: 'higher',
      description: '色彩饱和度增加',
    },
    {
      key: 'sharpnessGain',
      label: '清晰度提升',
      value: metricsSource.sharpnessGain,
      unit: '%',
      idealRange: '30%-60%',
      better: 'higher',
      description: '图像清晰度增加',
    },
  ];
}

function formatMetricValue(value: number, unit?: string): string {
  const formatted = value.toFixed(value < 1 ? 4 : 2);
  return unit ? `${formatted} ${unit}` : formatted;
}

function getBarColor(key: string, value: number): string {
  if (key === 'psnr') {
    if (value > 30) return theme.colors.status.success;
    if (value > 25) return theme.colors.status.warning;
    return theme.colors.status.error;
  }
  if (key === 'ssim') {
    if (value > 0.9) return theme.colors.status.success;
    if (value > 0.85) return theme.colors.status.warning;
    return theme.colors.status.error;
  }
  return theme.colors.primary;
}

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  statusCard: {
    margin: theme.spacing.md,
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    ...theme.layout.shadows.sm,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.xs,
  },
  statusTitle: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  statusDesc: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    marginBottom: theme.spacing.md,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  loadingText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.primary,
  },
  errorRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 4,
    backgroundColor: `${theme.colors.status.error}10`,
    padding: theme.spacing.sm,
    borderRadius: theme.layout.borderRadius.sm,
    marginBottom: theme.spacing.md,
  },
  errorText: {
    flex: 1,
    fontSize: theme.typography.sizes.small,
    color: theme.colors.status.error,
  },
  evaluateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.primary,
  },
  evaluateButtonDisabled: {
    opacity: 0.6,
  },
  evaluateButtonText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  section: {
    marginHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.md,
    padding: theme.spacing.lg,
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    ...theme.layout.shadows.sm,
  },
  sectionTitle: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
    marginBottom: theme.spacing.md,
  },
  metricItem: {
    marginBottom: theme.spacing.md,
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.xs,
  },
  metricLabelRow: {
    flex: 1,
    gap: 2,
  },
  metricLabel: {
    fontSize: theme.typography.sizes.bodySmall,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  metricIdeal: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  metricValue: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.bold,
    color: theme.colors.primary,
  },
  barContainer: {
    height: 8,
    backgroundColor: theme.colors.background.tertiary,
    borderRadius: theme.layout.borderRadius.full,
    overflow: 'hidden',
    marginBottom: theme.spacing.xs,
  },
  barFill: {
    height: '100%',
    borderRadius: theme.layout.borderRadius.full,
  },
  metricDesc: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  previewRow: {
    flexDirection: 'row',
    gap: theme.spacing.sm,
  },
  previewItem: {
    flex: 1,
    backgroundColor: theme.colors.background.secondary,
    borderRadius: theme.layout.borderRadius.md,
    overflow: 'hidden',
  },
  previewBadgeRow: {
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: theme.spacing.xs,
    backgroundColor: theme.colors.background.tertiary,
  },
  previewBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.layout.borderRadius.full,
  },
  previewBadgeOriginal: {
    backgroundColor: theme.colors.text.tertiary,
  },
  previewBadgeResult: {
    backgroundColor: theme.colors.status.success,
  },
  previewBadgeText: {
    fontSize: theme.typography.sizes.tiny,
    color: '#fff',
    fontWeight: theme.typography.weights.medium,
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

const metricsStyles = StyleSheet.create({
  previewImage: {
    width: '100%',
    height: 150,
  },
});

export default MetricsScreen;
