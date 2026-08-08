/**
 * 去雾 Tab (L1) — 页内步骤流
 *
 * 5 步流程：上传 → 算法选择 → 参数调节 → 处理 → 对比入口
 * 步骤间可自由回退，前置条件校验。
 * 复用现有 image-input / algorithm-select / processing 的核心逻辑。
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NavigationProp } from '@react-navigation/native';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { colors } from '@/theme/colors';
import { spacing, layout } from '@/theme/spacing';
import { AlgorithmAPI } from 'dehaze-sdk-js';
import type { Algorithm } from 'dehaze-sdk-js';
import type { RootStackParamList } from '@/routes/types';
import type { SelectedImage } from '@/types/image';
import type { CommonAlgorithmParams, ProcessingResult, TaskProgress } from '@/types/processing';
import { predictSingle, DEFAULT_PARAMS } from '@/pages/processing/services/processingApi';

type Step = 1 | 2 | 3 | 4 | 5;
type Phase = 'config' | 'processing' | 'done' | 'failed';

const STEPS = [
  { step: 1 as Step, label: '上传' },
  { step: 2 as Step, label: '算法' },
  { step: 3 as Step, label: '参数' },
  { step: 4 as Step, label: '处理' },
  { step: 5 as Step, label: '对比' },
];

export default function DehazeScreen() {
  const navigation = useNavigation<NavigationProp<RootStackParamList>>();

  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [algoLoading, setAlgoLoading] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);
  const [params, setParams] = useState<CommonAlgorithmParams>({ ...DEFAULT_PARAMS });
  const [phase, setPhase] = useState<Phase>('config');
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const cancelSignalRef = useRef<{ canceled: boolean }>({ canceled: false });

  useEffect(() => {
    if (currentStep === 2 && algorithms.length === 0) {
      setAlgoLoading(true);
      AlgorithmAPI.getList()
        .then(data => {
          const collect = (nodes: Algorithm[]): Algorithm[] => {
            const collected: Algorithm[] = [];
            for (const node of nodes) {
              if (!node.children || node.children.length === 0) collected.push(node);
              else collected.push(...collect(node.children));
            }
            return collected;
          };
          setAlgorithms(collect(data));
        })
        .catch(() => Alert.alert('加载失败', '无法加载算法列表，请稍后重试'))
        .finally(() => setAlgoLoading(false));
    }
  }, [currentStep, algorithms.length]);

  const goToStep = useCallback(
    (step: Step) => {
      if (step >= 2 && !selectedImage) { Alert.alert('提示', '请先上传图片'); return; }
      if (step >= 3 && !selectedAlgorithm) { Alert.alert('提示', '请先选择算法'); return; }
      if (step === 4) { handleStartProcessing(); return; }
      setCurrentStep(step);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedImage, selectedAlgorithm],
  );

  const handlePickImage = useCallback(() => {
    Alert.alert('选择图片', '请通过图像输入页选择图片，或使用样例图片', [
      { text: '取消', style: 'cancel' },
      {
        text: '使用样例',
        onPress: () => {
          setSelectedImage({ url: 'https://picsum.photos/800/600', thumbUrl: 'https://picsum.photos/200/150', name: '样例图片', source: 'sample' });
          setCurrentStep(2);
        },
      },
    ]);
  }, []);

  const handleSelectAlgorithm = useCallback((algo: Algorithm) => {
    setSelectedAlgorithm(algo);
    setCurrentStep(3);
  }, []);

  const handleParamChange = useCallback((key: keyof CommonAlgorithmParams, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const handleStartProcessing = useCallback(() => {
    if (!selectedImage?.url || !selectedAlgorithm?.id) { Alert.alert('提示', '缺少图片或算法信息'); return; }
    setCurrentStep(4);
    cancelSignalRef.current = { canceled: false };
    setPhase('processing');
    setProgress({ status: 0, elapsed: 0 });
    setResult(null);
    predictSingle({
      algorithmId: selectedAlgorithm.id,
      imageUrl: selectedImage.url,
      params,
      onProgress: p => setProgress(p),
      cancelSignal: cancelSignalRef.current,
    })
      .then(res => { setResult(res); setPhase('done'); setCurrentStep(5); })
      .catch(err => {
        const isCanceled = err instanceof Error && err.message.includes('取消');
        setProgress(prev => ({ status: isCanceled ? 4 : 3, elapsed: prev?.elapsed ?? 0, error: err instanceof Error ? err.message : '处理失败' }));
        setPhase(isCanceled ? 'config' : 'failed');
        if (isCanceled) setCurrentStep(3);
        else Alert.alert('处理失败', err instanceof Error ? err.message : '请稍后重试');
      });
  }, [selectedImage, selectedAlgorithm, params]);

  const handleCancel = useCallback(() => {
    Alert.alert('确认取消', '确定要取消当前处理任务吗？', [
      { text: '继续处理', style: 'cancel' },
      { text: '取消处理', style: 'destructive', onPress: () => { cancelSignalRef.current.canceled = true; } },
    ]);
  }, []);

  const handleEnterCompare = useCallback(() => {
    if (!selectedImage?.url || !result?.resultUrl) return;
    navigation.navigate('CompareSideBySide' as any, { originalUrl: selectedImage.url, processedUrl: result.resultUrl, algorithmId: selectedAlgorithm?.id });
  }, [selectedImage, result, selectedAlgorithm, navigation]);

  const formatTime = (ms: number) => `${(ms / 1000).toFixed(1)}s`;

  const renderStepIndicator = () => (
    <View style={styles.stepRow}>
      {STEPS.map((s, idx) => {
        const isActive = s.step === currentStep;
        const isDone = s.step < currentStep;
        return (
          <React.Fragment key={s.step}>
            <TouchableOpacity
              style={[styles.stepDot, isActive && styles.stepDotActive, isDone && styles.stepDotDone]}
              onPress={() => goToStep(s.step)}
              disabled={s.step > currentStep + 1}
            >
              {isDone ? <Ionicons name="checkmark" size={12} color="#fff" /> : <Text style={[styles.stepNum, (isActive || isDone) && styles.stepNumActive]}>{s.step}</Text>}
            </TouchableOpacity>
            <Text style={[styles.stepLabel, isActive && styles.stepLabelActive, isDone && styles.stepLabelDone]}>{s.label}</Text>
            {idx < STEPS.length - 1 && <View style={styles.stepLine} />}
          </React.Fragment>
        );
      })}
    </View>
  );

  const renderUpload = () => (
    <View style={styles.stepContent}>
      {selectedImage ? (
        <View style={styles.imageCard}>
          <Image source={{ uri: selectedImage.thumbUrl || selectedImage.url }} style={styles.previewImage} resizeMode="cover" />
          <View style={styles.imageInfo}>
            <Text style={styles.imageName} numberOfLines={1}>{selectedImage.name || '已选图片'}</Text>
            <TouchableOpacity onPress={() => { setSelectedImage(null); setSelectedAlgorithm(null); }}><Text style={styles.changeText}>重新选择</Text></TouchableOpacity>
          </View>
          <TouchableOpacity style={styles.nextBtn} onPress={() => goToStep(2)}><Text style={styles.nextBtnText}>下一步：选择算法</Text><Ionicons name="arrow-forward" size={16} color="#fff" /></TouchableOpacity>
        </View>
      ) : (
        <TouchableOpacity style={styles.uploadArea} onPress={handlePickImage} activeOpacity={0.8}>
          <Ionicons name="cloud-upload-outline" size={40} color={colors.primary} />
          <Text style={styles.uploadTitle}>点击上传图片</Text>
          <Text style={styles.uploadHint}>支持 JPG / PNG / GIF</Text>
        </TouchableOpacity>
      )}
    </View>
  );

  const renderAlgorithmSelect = () => (
    <View style={styles.stepContent}>
      {algoLoading ? (
        <ActivityIndicator size="large" color={colors.primary} style={styles.loading} />
      ) : (
        <>
          <Text style={styles.stepTitle}>选择算法</Text>
          {algorithms.slice(0, 8).map(algo => (
            <TouchableOpacity
              key={algo.id}
              style={[styles.algoCard, selectedAlgorithm?.id === algo.id && styles.algoCardSelected]}
              onPress={() => handleSelectAlgorithm(algo)}
              activeOpacity={0.7}
            >
              <View style={styles.algoInfo}>
                <Text style={styles.algoName}>{algo.name}</Text>
                <Text style={styles.algoDesc} numberOfLines={1}>{algo.description || `${algo.type ?? '通用类型'} · v${algo.version ?? '1.0'}`}</Text>
              </View>
              {selectedAlgorithm?.id === algo.id && <Ionicons name="checkmark-circle" size={22} color={colors.primary} />}
            </TouchableOpacity>
          ))}
          {algorithms.length === 0 && !algoLoading && <Text style={styles.emptyText}>暂无可用算法</Text>}
          <TouchableOpacity style={styles.backLink} onPress={() => setCurrentStep(1)}>
            <Ionicons name="arrow-back" size={14} color={colors.text.secondary} />
            <Text style={styles.backLinkText}>返回重新上传</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );

  const renderParams = () => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>参数调节</Text>
      {selectedAlgorithm && (
        <View style={styles.algoChip}>
          <Ionicons name="git-network-outline" size={14} color={colors.primary} />
          <Text style={styles.algoChipText}>{selectedAlgorithm.name}</Text>
          <TouchableOpacity onPress={() => setCurrentStep(2)}><Text style={styles.changeText}>更换</Text></TouchableOpacity>
        </View>
      )}
      {([
        { key: 'strength' as const, label: '去雾强度', min: 0, max: 100, step: 25, defaultVal: 50 },
        { key: 'saturation' as const, label: '色彩饱和度', min: 0, max: 200, step: 50, defaultVal: 100 },
        { key: 'contrast' as const, label: '对比度', min: 0, max: 200, step: 50, defaultVal: 100 },
        { key: 'sharpen' as const, label: '锐化程度', min: 0, max: 100, step: 25, defaultVal: 30 },
      ] as const).map(p => {
        const val = params[p.key] ?? p.defaultVal;
        const marks: number[] = [];
        for (let v = p.min; v <= p.max; v += p.step) marks.push(v);
        return (
          <View style={styles.paramGroup} key={p.key}>
            <View style={styles.paramHeader}>
              <Text style={styles.paramLabel}>{p.label}</Text>
              <Text style={styles.paramValue}>{val}</Text>
            </View>
            <View style={styles.sliderTrack}>
              <View style={[styles.sliderFill, { width: `${((val - p.min) / (p.max - p.min)) * 100}%` as any }]} />
            </View>
            <View style={styles.sliderBtns}>
              {marks.map(v => (
                <TouchableOpacity key={v} onPress={() => handleParamChange(p.key, v)}>
                  <Text style={[styles.sliderBtnText, val === v && styles.sliderBtnActive]}>{v}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        );
      })}
      <TouchableOpacity style={styles.startBtn} onPress={() => goToStep(4)} activeOpacity={0.8}>
        <Ionicons name="flash" size={18} color="#fff" />
        <Text style={styles.startBtnText}>开始去雾</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.backLink} onPress={() => setCurrentStep(2)}>
        <Ionicons name="arrow-back" size={14} color={colors.text.secondary} />
        <Text style={styles.backLinkText}>返回选择算法</Text>
      </TouchableOpacity>
    </View>
  );

  const renderProcessing = () => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>处理中</Text>
      {progress && (
        <View style={styles.progressCard}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.progressText}>{progress.status === 2 ? '处理完成' : progress.status === 3 ? '处理失败' : '正在处理...'}</Text>
          {progress.elapsed !== undefined && <Text style={styles.progressTime}>已用 {formatTime(progress.elapsed)}</Text>}
          {progress.error && <Text style={styles.progressError}>{progress.error}</Text>}
        </View>
      )}
      {phase === 'processing' && (
        <TouchableOpacity style={styles.cancelBtn} onPress={handleCancel}><Text style={styles.cancelBtnText}>取消处理</Text></TouchableOpacity>
      )}
      {phase === 'failed' && (
        <TouchableOpacity style={styles.startBtn} onPress={handleStartProcessing}><Text style={styles.startBtnText}>重新处理</Text></TouchableOpacity>
      )}
    </View>
  );

  const renderResult = () => (
    <View style={styles.stepContent}>
      <Text style={styles.stepTitle}>处理完成</Text>
      {result && selectedImage && (
        <View style={styles.resultCard}>
          <View style={styles.resultImages}>
            <View style={styles.resultImgWrap}>
              <Image source={{ uri: selectedImage.thumbUrl || selectedImage.url }} style={styles.resultImg} resizeMode="cover" />
              <Text style={styles.resultImgLabel}>原图</Text>
            </View>
            <Ionicons name="arrow-forward" size={20} color={colors.text.tertiary} />
            <View style={styles.resultImgWrap}>
              <Image source={{ uri: result.resultThumbnailUrl || result.resultUrl }} style={styles.resultImg} resizeMode="cover" />
              <Text style={styles.resultImgLabel}>结果</Text>
            </View>
          </View>
          <Text style={styles.resultMeta}>算法：{selectedAlgorithm?.name ?? '未知'} · 耗时：{result.time ? formatTime(result.time) : '--'}</Text>
          <TouchableOpacity style={styles.compareBtn} onPress={handleEnterCompare} activeOpacity={0.8}>
            <Ionicons name="git-compare-outline" size={18} color="#fff" />
            <Text style={styles.compareBtnText}>进入效果对比</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.reprocessBtn} onPress={() => { setResult(null); setPhase('config'); setCurrentStep(3); }}>
            <Text style={styles.reprocessBtnText}>重新处理</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1: return renderUpload();
      case 2: return renderAlgorithmSelect();
      case 3: return renderParams();
      case 4: return renderProcessing();
      case 5: return renderResult();
      default: return null;
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>去雾处理</Text>
        </View>
        {renderStepIndicator()}
        <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          {renderCurrentStep()}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background.secondary },
  container: { flex: 1 },
  header: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm },
  headerTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary },
  scroll: { flex: 1 },
  scrollContent: { paddingBottom: spacing.xxxl },
  stepRow: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.md, paddingVertical: spacing.sm, backgroundColor: colors.background.primary, marginBottom: spacing.sm, gap: 4 },
  stepDot: { width: 24, height: 24, borderRadius: 12, backgroundColor: colors.background.tertiary, justifyContent: 'center', alignItems: 'center' },
  stepDotActive: { backgroundColor: colors.primary },
  stepDotDone: { backgroundColor: colors.status.success },
  stepNum: { fontSize: 11, fontWeight: '600', color: colors.text.tertiary },
  stepNumActive: { color: '#fff' },
  stepLabel: { fontSize: 10, color: colors.text.tertiary },
  stepLabelActive: { color: colors.primary, fontWeight: '600' },
  stepLabelDone: { color: colors.status.success },
  stepLine: { flex: 1, height: 1, backgroundColor: colors.border.light, marginHorizontal: 2 },
  stepContent: { paddingHorizontal: spacing.md, paddingTop: spacing.md },
  stepTitle: { fontSize: 16, fontWeight: '600', color: colors.text.primary, marginBottom: spacing.md },
  uploadArea: { borderWidth: 2, borderColor: colors.border.light, borderStyle: 'dashed', borderRadius: layout.borderRadius.lg, paddingVertical: spacing.xxxl, alignItems: 'center', gap: spacing.sm },
  uploadTitle: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
  uploadHint: { fontSize: 13, color: colors.text.tertiary },
  imageCard: { backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.lg, padding: spacing.md, ...layout.shadows.sm },
  previewImage: { width: '100%', height: 200, borderRadius: layout.borderRadius.md, backgroundColor: colors.background.tertiary },
  imageInfo: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: spacing.sm },
  imageName: { flex: 1, fontSize: 14, color: colors.text.primary, fontWeight: '500' },
  changeText: { fontSize: 13, color: colors.primary, fontWeight: '500' },
  nextBtn: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: spacing.sm, marginTop: spacing.md, paddingVertical: spacing.md, backgroundColor: colors.primary, borderRadius: layout.borderRadius.md },
  nextBtnText: { fontSize: 15, fontWeight: '600', color: '#fff' },
  loading: { marginTop: spacing.xxxl },
  emptyText: { textAlign: 'center', color: colors.text.tertiary, marginTop: spacing.xl },
  algoCard: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, marginBottom: spacing.sm, backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.md, borderWidth: 1.5, borderColor: 'transparent', ...layout.shadows.sm },
  algoCardSelected: { borderColor: colors.primary, backgroundColor: colors.primaryLight },
  algoInfo: { flex: 1, gap: 2 },
  algoName: { fontSize: 15, fontWeight: '600', color: colors.text.primary },
  algoDesc: { fontSize: 12, color: colors.text.secondary },
  algoChip: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, paddingHorizontal: spacing.md, paddingVertical: spacing.sm, backgroundColor: colors.primaryLight, borderRadius: layout.borderRadius.full, alignSelf: 'flex-start', marginBottom: spacing.lg },
  algoChipText: { fontSize: 13, fontWeight: '500', color: colors.primary },
  paramGroup: { marginBottom: spacing.lg },
  paramHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: spacing.sm },
  paramLabel: { fontSize: 14, fontWeight: '500', color: colors.text.primary },
  paramValue: { fontSize: 14, fontWeight: '600', color: colors.primary },
  sliderTrack: { height: 6, backgroundColor: colors.background.tertiary, borderRadius: 3, marginBottom: spacing.sm, overflow: 'hidden' },
  sliderFill: { height: '100%', backgroundColor: colors.primary, borderRadius: 3 },
  sliderBtns: { flexDirection: 'row', justifyContent: 'space-between' },
  sliderBtnText: { fontSize: 12, color: colors.text.tertiary },
  sliderBtnActive: { color: colors.primary, fontWeight: '600' },
  startBtn: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.md, backgroundColor: colors.primary, borderRadius: layout.borderRadius.md, marginTop: spacing.md },
  startBtnText: { fontSize: 16, fontWeight: '700', color: '#fff' },
  backLink: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: spacing.xs, marginTop: spacing.md, paddingVertical: spacing.sm },
  backLinkText: { fontSize: 13, color: colors.text.secondary },
  progressCard: { alignItems: 'center', padding: spacing.xl, backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.lg, ...layout.shadows.sm },
  progressText: { fontSize: 16, fontWeight: '600', color: colors.text.primary, marginTop: spacing.md },
  progressTime: { fontSize: 13, color: colors.text.secondary, marginTop: spacing.xs },
  progressError: { fontSize: 13, color: colors.status.error, marginTop: spacing.sm },
  cancelBtn: { alignItems: 'center', paddingVertical: spacing.md, marginTop: spacing.md, borderRadius: layout.borderRadius.md, borderWidth: 1, borderColor: colors.status.error },
  cancelBtnText: { fontSize: 14, fontWeight: '600', color: colors.status.error },
  resultCard: { backgroundColor: colors.background.primary, borderRadius: layout.borderRadius.lg, padding: spacing.lg, ...layout.shadows.sm },
  resultImages: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.md, marginBottom: spacing.md },
  resultImgWrap: { alignItems: 'center', gap: spacing.xs },
  resultImg: { width: 120, height: 90, borderRadius: layout.borderRadius.md, backgroundColor: colors.background.tertiary },
  resultImgLabel: { fontSize: 11, color: colors.text.tertiary },
  resultMeta: { fontSize: 13, color: colors.text.secondary, textAlign: 'center', marginBottom: spacing.lg },
  compareBtn: { flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.md, backgroundColor: colors.primary, borderRadius: layout.borderRadius.md },
  compareBtnText: { fontSize: 15, fontWeight: '600', color: '#fff' },
  reprocessBtn: { alignItems: 'center', paddingVertical: spacing.md, marginTop: spacing.sm },
  reprocessBtnText: { fontSize: 14, color: colors.text.secondary },
});
