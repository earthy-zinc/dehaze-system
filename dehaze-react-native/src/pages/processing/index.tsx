/**
 * 去雾处理页面
 *
 * 业务流程：
 * 1. 从算法选择页接收 { algorithmId, image }
 * 2. 加载算法详情（AlgorithmAPI.getAlgorithmInfoById）
 * 3. 显示原图预览 + 算法信息
 * 4. 提供参数调节面板（可选展开）
 * 5. 用户点击「开始去雾」→ 确认对话框 → 调用 predictSingle
 * 6. 处理过程中显示真实状态与已用时间（ProcessingProgress，API 同步返回不模拟进度）
 * 7. 处理完成显示结果预览（ResultPreview），可进入效果对比或重新处理
 * 8. 支持取消处理
 */
import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '@/routes/types';
import { MainLayout } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import ImageLoader from '@/components/ImageLoader';
import { AlgorithmAPI } from 'dehaze-sdk-js';
import type { Algorithm } from '@/types/algorithm';
import type {
  CommonAlgorithmParams,
  ProcessingResult,
  TaskProgress,
} from '@/types/processing';
import { useResponsive } from '@/hooks/useResponsive';
import {
  predictSingle,
  DEFAULT_PARAMS,
} from './services/processingApi';
import { historyStorage } from '@/pages/image-input/services/historyStorage';
import ProcessingProgress from './components/ProcessingProgress';
import ParamsPanel from './components/ParamsPanel';
import ResultPreview from './components/ResultPreview';

type Props = NativeStackScreenProps<RootStackParamList, 'Processing'>;

type Phase = 'config' | 'processing' | 'done' | 'failed';

const ProcessingScreen: React.FC<Props> = ({ route, navigation }) => {
  const { containerPadding } = useResponsive();
  const { algorithmId, image } = route.params ?? {};

  // 算法详情
  const [algorithm, setAlgorithm] = useState<Algorithm | null>(null);
  const [algorithmLoading, setAlgorithmLoading] = useState(false);

  // 参数（历史记录复用时从 image.algorithmParams 初始化）
  const [params, setParams] = useState<CommonAlgorithmParams>(() => {
    if (image?.algorithmParams) {
      try {
        const parsed = JSON.parse(image.algorithmParams) as CommonAlgorithmParams;
        return { ...DEFAULT_PARAMS, ...parsed };
      } catch {
        // 参数解析失败时回退到默认值
      }
    }
    return { ...DEFAULT_PARAMS };
  });
  const [showParams, setShowParams] = useState(false);

  // 处理状态
  const [phase, setPhase] = useState<Phase>('config');
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);

  // 取消信号（使用 ref 避免闭包陈旧）
  const cancelSignalRef = useRef<{ canceled: boolean }>({ canceled: false });

  // 加载算法详情
  useEffect(() => {
    if (!algorithmId) return;
    setAlgorithmLoading(true);
    AlgorithmAPI.getAlgorithmInfoById(algorithmId)
      .then(data => setAlgorithm(data))
      .catch(err => {
        Alert.alert('加载算法失败', err instanceof Error ? err.message : '请稍后重试');
      })
      .finally(() => setAlgorithmLoading(false));
  }, [algorithmId]);

  /** 开始去雾处理（实际执行） */
  const startProcessing = useCallback(() => {
    if (!image?.url || !algorithmId) return;

    // 重置状态
    cancelSignalRef.current = { canceled: false };
    setPhase('processing');
    setProgress({
      status: 'idle',
      elapsed: 0,
    });
    setResult(null);

    predictSingle({
      algorithmId,
      imageUrl: image.url,
      params,
      onProgress: p => {
        setProgress(p);
      },
      cancelSignal: cancelSignalRef.current,
    })
      .then(res => {
        setResult(res);
        setPhase('done');

        // 写入图像输入历史记录（失败不阻塞主流程）
        historyStorage
          .addRecord({
            originalImageUrl: image.url,
            originalThumbnailUrl: image.thumbUrl,
            resultImageUrl: res.resultUrl,
            resultThumbnailUrl: res.resultThumbnailUrl,
            algorithmId,
            algorithmName: algorithm?.name,
            algorithmParams: JSON.stringify(params),
            processingTime: res.time,
            status: 1,
            inputSource: image.source,
          })
          .catch(() => {
            /* 历史记录写入失败不影响处理结果展示 */
          });
      })
      .catch(err => {
        const isCanceled = err instanceof Error && err.message.includes('取消');
        setProgress(prev => ({
          status: isCanceled ? 'canceled' : 'failed',
          elapsed: prev?.elapsed ?? 0,
          error: err instanceof Error ? err.message : '处理失败',
        }));
        setPhase(isCanceled ? 'config' : 'failed');
        if (!isCanceled) {
          Alert.alert('处理失败', err instanceof Error ? err.message : '请稍后重试');
        }
      });
  }, [image, algorithmId, algorithm, params]);

  /** 开始去雾处理（弹出确认对话框） */
  const handleStart = useCallback(() => {
    if (!image?.url || !algorithmId) {
      Alert.alert('提示', '缺少图片或算法信息，请返回上一步重新选择');
      return;
    }

    Alert.alert(
      '确认开始去雾',
      `图片：${image.name ?? '未命名'}\n算法：${algorithm?.name ?? algorithmId}\n参数：去雾强度 ${params.strength ?? 50}`,
      [
        { text: '取消', style: 'cancel' },
        {
          text: '开始处理',
          onPress: () => {
            startProcessing();
          },
        },
      ],
    );
  }, [image, algorithmId, algorithm, params, startProcessing]);

  /** 取消处理 */
  const handleCancel = useCallback(() => {
    Alert.alert('确认取消', '确定要取消当前处理任务吗？', [
      { text: '继续处理', style: 'cancel' },
      {
        text: '取消处理',
        style: 'destructive',
        onPress: () => {
          cancelSignalRef.current.canceled = true;
        },
      },
    ]);
  }, []);

  /** 重新处理 */
  const handleReprocess = useCallback(() => {
    setResult(null);
    setProgress(null);
    setPhase('config');
  }, []);

  /** 进入效果对比 */
  const handleEnterCompare = useCallback(() => {
    if (!image?.url || !result?.resultUrl) return;
    // 默认进入并排对比（携带 algorithmId 与 GT 参考图 cleanUrl，供指标评估使用）
    navigation.navigate('SideBySide', {
      originalUrl: image.url,
      processedUrl: result.resultUrl,
      cleanUrl: image.cleanUrl,
      algorithmId,
    });
  }, [image, result, navigation, algorithmId]);

  /** 返回图像输入页 */
  const handleBackToImageInput = useCallback(() => {
    navigation.navigate('ImageInput');
  }, [navigation]);

  /** 返回算法选择页 */
  const handleBackToAlgorithmSelect = useCallback(() => {
    if (image) {
      navigation.navigate('AlgorithmSelect', { image });
    } else {
      navigation.navigate('AlgorithmSelect');
    }
  }, [navigation, image]);

  // 缺少图片或算法参数
  if (!image?.url || !algorithmId) {
    return (
      <MainLayout title="图像处理" showBack>
        <View style={styles.emptyContainer}>
          <Icon name="image" size={48} color={theme.colors.text.tertiary} />
          <Text style={styles.emptyTitle}>缺少必要的图片或算法信息</Text>
          <Text style={styles.emptyDesc}>请先选择图片与算法</Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={handleBackToImageInput}
          >
            <Text style={styles.emptyButtonText}>去选择图片</Text>
          </TouchableOpacity>
        </View>
      </MainLayout>
    );
  }

  return (
    <MainLayout title="图像处理" showBack>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          { padding: containerPadding },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* 算法与图片信息卡片 */}
        <View style={styles.infoCard}>
          <View style={styles.infoHeader}>
            <View style={styles.infoIconWrapper}>
              <Icon name="brain" size={20} color={theme.colors.primary} />
            </View>
            <View style={styles.infoHeaderText}>
              <Text style={styles.infoTitle} numberOfLines={1}>
                {algorithmLoading ? '加载算法中...' : algorithm?.name ?? '未知算法'}
              </Text>
              <Text style={styles.infoSubtitle}>
                {algorithm
                  ? `${algorithm.type ?? '通用类型'} · v${algorithm.version ?? '1.0'}`
                  : `算法ID: ${algorithmId}`}
              </Text>
            </View>
            {algorithmLoading && (
              <ActivityIndicator size="small" color={theme.colors.primary} />
            )}
          </View>

          {/* 图片缩略图 */}
          <View style={styles.imagePreviewRow}>
            <View style={styles.imagePreviewItem}>
              <ImageLoader
                source={{ uri: image.thumbUrl || image.url }}
                style={styles.imageThumb}
                resizeMode="cover"
              />
              <Text style={styles.imageLabel}>待处理图片</Text>
              {image.name && (
                <Text style={styles.imageName} numberOfLines={1}>
                  {image.name}
                </Text>
              )}
            </View>
            <View style={styles.imagePreviewArrow}>
              <Icon name="arrow-right" size={20} color={theme.colors.text.tertiary} />
            </View>
            <View style={styles.imagePreviewItem}>
              <View style={[styles.imageThumb, styles.imageThumbPlaceholder]}>
                {result && (result.resultThumbnailUrl || result.resultUrl) ? (
                  <ImageLoader
                    source={{ uri: result.resultThumbnailUrl || result.resultUrl }}
                    style={styles.imageThumb}
                    resizeMode="cover"
                  />
                ) : (
                  <Icon
                    name="image"
                    size={32}
                    color={theme.colors.text.tertiary}
                  />
                )}
              </View>
              <Text style={styles.imageLabel}>处理结果</Text>
              <Text style={styles.imageName}>
                {result ? '已完成' : phase === 'processing' ? '处理中...' : '待处理'}
              </Text>
            </View>
          </View>
        </View>

        {/* 配置阶段：参数调节 + 开始按钮 */}
        {phase === 'config' && (
          <>
            {/* 参数调节（可折叠） */}
            <TouchableOpacity
              style={styles.collapseHeader}
              onPress={() => setShowParams(v => !v)}
              activeOpacity={0.7}
            >
              <View style={styles.collapseHeaderLeft}>
                <Icon name="settings" size={16} color={theme.colors.text.secondary} />
                <Text style={styles.collapseHeaderText}>参数调节</Text>
              </View>
              <Icon
                name={showParams ? 'arrow-down' : 'arrow-right'}
                size={14}
                color={theme.colors.text.tertiary}
              />
            </TouchableOpacity>

            {showParams && (
              <View style={styles.collapseBody}>
                <ParamsPanel
                  params={params}
                  onChange={setParams}
                  disabled={phase !== 'config'}
                />
              </View>
            )}

            {/* 切换算法 */}
            <TouchableOpacity
              style={styles.changeAlgorithmButton}
              onPress={handleBackToAlgorithmSelect}
            >
              <Icon name="refresh" size={14} color={theme.colors.text.secondary} />
              <Text style={styles.changeAlgorithmText}>更换算法</Text>
            </TouchableOpacity>

            {/* 开始处理按钮 */}
            <TouchableOpacity
              style={styles.startButton}
              onPress={handleStart}
              activeOpacity={0.8}
            >
              <Icon name="bolt" size={18} color="#fff" />
              <Text style={styles.startButtonText}>开始去雾</Text>
            </TouchableOpacity>
          </>
        )}

        {/* 处理中：进度 */}
        {phase === 'processing' && progress && (
          <View>
            <ProcessingProgress progress={progress} />
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={handleCancel}
              activeOpacity={0.8}
            >
              <Icon name="times" size={16} color={theme.colors.status.error} />
              <Text style={styles.cancelButtonText}>取消处理</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 处理失败：错误 + 重试 */}
        {phase === 'failed' && progress && (
          <View>
            <ProcessingProgress progress={progress} />
            <TouchableOpacity
              style={styles.startButton}
              onPress={startProcessing}
              activeOpacity={0.8}
            >
              <Icon name="refresh" size={18} color="#fff" />
              <Text style={styles.startButtonText}>重新处理</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 处理完成：结果预览 */}
        {phase === 'done' && result && (
          <ResultPreview
            originalUrl={image.url}
            result={result}
            onEnterCompare={handleEnterCompare}
            onReprocess={handleReprocess}
          />
        )}
      </ScrollView>
    </MainLayout>
  );
};

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: theme.spacing.xxxl,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.xl,
    gap: theme.spacing.md,
  },
  emptyTitle: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  emptyDesc: {
    fontSize: theme.typography.sizes.body,
    color: theme.colors.text.secondary,
  },
  emptyButton: {
    marginTop: theme.spacing.sm,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.primary,
  },
  emptyButtonText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.semibold,
    color: '#fff',
  },
  infoCard: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.lg,
    padding: theme.spacing.lg,
    marginBottom: theme.spacing.md,
    ...theme.layout.shadows.sm,
  },
  infoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  infoIconWrapper: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: `${theme.colors.primary}15`,
    justifyContent: 'center',
    alignItems: 'center',
  },
  infoHeaderText: {
    flex: 1,
    gap: 2,
  },
  infoTitle: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.semibold,
    color: theme.colors.text.primary,
  },
  infoSubtitle: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  imagePreviewRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  imagePreviewItem: {
    flex: 1,
    gap: 4,
  },
  imagePreviewArrow: {
    paddingHorizontal: theme.spacing.xs,
  },
  imageThumb: {
    width: '100%',
    height: 100,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.background.tertiary,
  },
  imageThumbPlaceholder: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  imageLabel: {
    fontSize: theme.typography.sizes.tiny,
    color: theme.colors.text.tertiary,
  },
  imageName: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
    fontWeight: theme.typography.weights.medium,
  },
  collapseHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.layout.borderRadius.md,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  collapseHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  collapseHeaderText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.text.primary,
  },
  collapseBody: {
    marginBottom: theme.spacing.sm,
  },
  changeAlgorithmButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    paddingVertical: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  changeAlgorithmText: {
    fontSize: theme.typography.sizes.small,
    color: theme.colors.text.secondary,
  },
  startButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.sm,
    paddingVertical: theme.spacing.lg,
    borderRadius: theme.layout.borderRadius.md,
    backgroundColor: theme.colors.primary,
    ...theme.layout.shadows.sm,
  },
  startButtonText: {
    fontSize: theme.typography.sizes.bodyLarge,
    fontWeight: theme.typography.weights.bold,
    color: '#fff',
  },
  cancelButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.spacing.xs,
    paddingVertical: theme.spacing.md,
    marginTop: theme.spacing.md,
    borderRadius: theme.layout.borderRadius.md,
    borderWidth: 1,
    borderColor: theme.colors.status.error,
  },
  cancelButtonText: {
    fontSize: theme.typography.sizes.body,
    fontWeight: theme.typography.weights.medium,
    color: theme.colors.status.error,
  },
});

export default ProcessingScreen;
