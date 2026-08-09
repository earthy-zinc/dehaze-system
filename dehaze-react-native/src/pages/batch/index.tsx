/**
 * 批量处理页面（L2，工具 Tab / 去雾 Tab 均可访问）
 *
 * 对应 05-菜单与页面层级规划：
 * - 批量上传（最多 20 张，FlatList 网格展示 + 移除）
 * - 算法选择（仅展示已发布算法，选中高亮）
 * - 可选 JSON 参数
 * - 批量处理：ModelAPI.batchPredict + 未完成任务 predictAndWait 轮询
 * - 处理进度条（实时百分比）
 * - 结果列表（完成/失败状态，查看结果/重试）
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import { CompositeScreenProps } from '@react-navigation/native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { launchImageLibrary } from 'react-native-image-picker';
import type { ToolsStackParamList, DehazeStackParamList } from '@/routes/types';
import { AppHeader } from '@/layout';
import { theme } from '@/theme';
import Icon from '@/components/Icon';
import { AlgorithmAPI, ModelAPI } from 'dehaze-sdk-js';
import type { Algorithm, PredictionResultVO } from 'dehaze-sdk-js';

type Props = CompositeScreenProps<
  NativeStackScreenProps<ToolsStackParamList, 'Batch'>,
  NativeStackScreenProps<DehazeStackParamList, 'Batch'>
>;

interface BatchItem {
  id: string;
  uri: string;
  fileName: string;
  result?: PredictionResultVO;
  status: 'idle' | 'processing' | 'done' | 'failed';
  error?: string;
}

const MAX_IMAGES = 20;

const BatchScreen: React.FC<Props> = ({ navigation }) => {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [algLoading, setAlgLoading] = useState(true);
  const [selectedAlgId, setSelectedAlgId] = useState<number | null>(null);
  const [paramsText, setParamsText] = useState('{}');
  const [paramsError, setParamsError] = useState<string | null>(null);
  const [batchPhase, setBatchPhase] = useState<'config' | 'processing' | 'done'>('config');
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  const cancelRef = useRef(false);

  /** 加载已发布算法 */
  useEffect(() => {
    setAlgLoading(true);
    AlgorithmAPI.tree()
      .then(data => {
        const flatten = (nodes: Algorithm[]): Algorithm[] => {
          const result: Algorithm[] = [];
          for (const node of nodes) {
            if (node.children && node.children.length > 0) {
              result.push(...flatten(node.children));
            } else {
              result.push(node);
            }
          }
          return result;
        };
        setAlgorithms(flatten((data || []) as unknown as Algorithm[]));
      })
      .catch(() => Alert.alert('错误', '加载算法列表失败'))
      .finally(() => setAlgLoading(false));
  }, []);

  /** 选择图片 */
  const handlePickImages = useCallback(async () => {
    try {
      const result = await launchImageLibrary({
        selectionLimit: MAX_IMAGES - items.length,
        mediaType: 'photo',
        includeBase64: false,
      });

      if (result.didCancel) return;

      const newItems: BatchItem[] = (result.assets || [])
        .filter(a => a.uri)
        .map((a, i) => ({
          id: `img_${Date.now()}_${i}`,
          uri: a.uri!,
          fileName: a.fileName || `图片_${i + 1}`,
          status: 'idle' as const,
        }));

      setItems(prev => {
        const combined = [...prev, ...newItems];
        if (combined.length > MAX_IMAGES) {
          Alert.alert('提示', `最多上传 ${MAX_IMAGES} 张图片`);
          return combined.slice(0, MAX_IMAGES);
        }
        return combined;
      });
    } catch (err: unknown) {
      Alert.alert('错误', err instanceof Error ? err.message : '选择图片失败');
    }
  }, [items.length]);

  const handleRemoveItem = useCallback((id: string) => {
    setItems(prev => prev.filter(item => item.id !== id));
  }, []);

  /** 开始批量处理 */
  const handleStartBatch = useCallback(async () => {
    if (items.length === 0) { Alert.alert('提示', '请先选择图片'); return; }
    if (!selectedAlgId) { Alert.alert('提示', '请先选择算法'); return; }

    let params: Record<string, unknown> | null = null;
    try {
      params = JSON.parse(paramsText);
    } catch {
      setParamsError('JSON 格式无效');
      return;
    }
    setParamsError(null);

    cancelRef.current = false;
    setBatchPhase('processing');
    setProgress({ done: 0, total: items.length });
    setItems(prev => prev.map(item => ({ ...item, status: 'processing' as const })));

    for (let i = 0; i < items.length; i++) {
      if (cancelRef.current) break;
      const item = items[i];
      try {
        const result = await ModelAPI.predictAndWait({
          algorithmId: selectedAlgId,
          imageUrl: item.uri,
          params: params ? JSON.stringify(params) : undefined,
        });

        if (result.status === 2) {
          setItems(prev =>
            prev.map(it => (it.id === item.id ? { ...it, status: 'done' as const, result } : it)),
          );
        } else {
          setItems(prev =>
            prev.map(it => (it.id === item.id ? { ...it, status: 'failed' as const, error: result.errorMessage || '处理失败' } : it)),
          );
        }
      } catch (err: unknown) {
        setItems(prev =>
          prev.map(it => (it.id === item.id ? { ...it, status: 'failed' as const, error: err instanceof Error ? err.message : '处理失败' } : it)),
        );
      }
      setProgress({ done: i + 1, total: items.length });
    }
    setBatchPhase('done');
  }, [items, selectedAlgId, paramsText]);

  const handleCancel = useCallback(() => {
    Alert.alert('确认取消', '确定要取消批量处理吗？', [
      { text: '继续处理', style: 'cancel' },
      { text: '取消', style: 'destructive', onPress: () => { cancelRef.current = true; setBatchPhase('done'); } },
    ]);
  }, []);

  const handleRetryItem = useCallback(async (item: BatchItem) => {
    if (!selectedAlgId) return;
    setItems(prev => prev.map(it => (it.id === item.id ? { ...it, status: 'processing' as const } : it)));
    try {
      const result = await ModelAPI.predictAndWait({
        algorithmId: selectedAlgId,
        imageUrl: item.uri,
        params: paramsText !== '{}' ? paramsText : undefined,
      });
      if (result.status === 2) {
        setItems(prev => prev.map(it => (it.id === item.id ? { ...it, status: 'done' as const, result } : it)));
      } else {
        setItems(prev => prev.map(it => (it.id === item.id ? { ...it, status: 'failed' as const, error: result.errorMessage || '处理失败' } : it)));
      }
    } catch (err: unknown) {
      setItems(prev => prev.map(it => (it.id === item.id ? { ...it, status: 'failed' as const, error: err instanceof Error ? err.message : '处理失败' } : it)));
    }
  }, [selectedAlgId, paramsText]);

  const handleReset = useCallback(() => {
    setItems([]);
    setSelectedAlgId(null);
    setParamsText('{}');
    setParamsError(null);
    setBatchPhase('config');
    setProgress({ done: 0, total: 0 });
  }, []);

  const handleViewResult = useCallback(
    (item: BatchItem) => {
      if (!item.result?.resultUrl) return;
      navigation.navigate('CompareSideBySide', {
        originalUrl: item.uri,
        processedUrl: item.result.resultUrl,
        algorithmId: selectedAlgId ?? undefined,
      });
    },
    [navigation, selectedAlgId],
  );

  /** 渲染图片项 */
  const renderImageItem = (item: BatchItem) => (
    <View key={item.id} style={styles.imageItem}>
      <Image source={{ uri: item.uri }} style={styles.thumbImage} />
      <TouchableOpacity style={styles.removeBtn} onPress={() => handleRemoveItem(item.id)} hitSlop={8}>
        <Icon name="times" size={14} color="#fff" />
      </TouchableOpacity>
      {item.status !== 'idle' && (
        <View style={[styles.statusBadge, item.status === 'done' ? styles.statusDone : item.status === 'failed' ? styles.statusFailed : styles.statusProcessing]}>
          <Text style={styles.statusBadgeText}>
            {item.status === 'done' ? '完成' : item.status === 'failed' ? '失败' : '处理中'}
          </Text>
        </View>
      )}
      {item.status === 'done' && (
        <TouchableOpacity style={styles.viewResultBtn} onPress={() => handleViewResult(item)} activeOpacity={0.7}>
          <Icon name="eye" size={12} color="#fff" />
          <Text style={styles.viewResultText}>对比</Text>
        </TouchableOpacity>
      )}
      {item.status === 'failed' && (
        <TouchableOpacity style={styles.retryBtn} onPress={() => handleRetryItem(item)} activeOpacity={0.7}>
          <Icon name="refresh" size={12} color="#fff" />
          <Text style={styles.viewResultText}>重试</Text>
        </TouchableOpacity>
      )}
    </View>
  );

  const doneCount = items.filter(i => i.status === 'done').length;
  const failedCount = items.filter(i => i.status === 'failed').length;

  return (
    <View style={styles.container}>
      <AppHeader title="批量处理" showBack onBackPress={() => navigation.goBack()} />
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* 图片上传 */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>待处理图片 ({items.length}/{MAX_IMAGES})</Text>
            {items.length < MAX_IMAGES && batchPhase === 'config' && (
              <TouchableOpacity style={styles.addBtn} onPress={handlePickImages} activeOpacity={0.7}>
                <Icon name="plus" size={14} color={theme.colors.primary} />
                <Text style={styles.addBtnText}>添加图片</Text>
              </TouchableOpacity>
            )}
          </View>
          {items.length > 0 ? (
            <View style={styles.imageGrid}>{items.map(renderImageItem)}</View>
          ) : (
            <TouchableOpacity style={styles.uploadPlaceholder} onPress={handlePickImages} activeOpacity={0.7}>
              <Icon name="upload" size={32} color={theme.colors.text.tertiary} />
              <Text style={styles.uploadPlaceholderText}>点击选择图片（最多 {MAX_IMAGES} 张）</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* 算法选择 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>选择算法</Text>
          {algLoading ? (
            <ActivityIndicator size="small" color={theme.colors.primary} />
          ) : (
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.algScrollContent}>
              {algorithms.map(alg => {
                const isSelected = selectedAlgId === alg.id;
                return (
                  <TouchableOpacity
                    key={alg.id}
                    style={[styles.algChip, isSelected && styles.algChipSelected]}
                    onPress={() => setSelectedAlgId(alg.id)}
                    activeOpacity={0.7}
                  >
                    <Text style={[styles.algChipText, isSelected && styles.algChipTextSelected]} numberOfLines={1}>
                      {alg.name}
                    </Text>
                  </TouchableOpacity>
                );
              })}
              {algorithms.length === 0 && !algLoading && (
                <Text style={styles.emptyAlgText}>暂无可用的已发布算法</Text>
              )}
            </ScrollView>
          )}
        </View>

        {/* 参数 */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>处理参数（JSON，可选）</Text>
          <TextInput
            style={[styles.paramsInput, paramsError && styles.paramsInputError]}
            value={paramsText}
            onChangeText={t => { setParamsText(t); setParamsError(null); }}
            multiline
            numberOfLines={4}
            placeholder='{"strength": 50}'
            placeholderTextColor={theme.colors.text.tertiary}
            autoCorrect={false}
            autoCapitalize="none"
          />
          {paramsError && <Text style={styles.paramsErrorText}>{paramsError}</Text>}
        </View>

        {/* 进度 */}
        {batchPhase === 'processing' && (
          <View style={styles.progressSection}>
            <View style={styles.progressHeader}>
              <Text style={styles.progressTitle}>批量处理中</Text>
              <Text style={styles.progressText}>{progress.done}/{progress.total}</Text>
            </View>
            <View style={styles.progressTrack}>
              <View style={[styles.progressFill, { width: `${progress.total > 0 ? (progress.done / progress.total) * 100 : 0}%` }]} />
            </View>
            <TouchableOpacity style={styles.cancelBtn} onPress={handleCancel} activeOpacity={0.7}>
              <Text style={styles.cancelBtnText}>取消处理</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 结果统计 */}
        {batchPhase === 'done' && items.length > 0 && (
          <View style={styles.resultSummary}>
            <View style={styles.resultStat}>
              <Text style={[styles.resultStatValue, { color: theme.colors.status.success }]}>{doneCount}</Text>
              <Text style={styles.resultStatLabel}>成功</Text>
            </View>
            <View style={styles.resultStatDivider} />
            <View style={styles.resultStat}>
              <Text style={[styles.resultStatValue, { color: theme.colors.status.error }]}>{failedCount}</Text>
              <Text style={styles.resultStatLabel}>失败</Text>
            </View>
            <View style={styles.resultStatDivider} />
            <View style={styles.resultStat}>
              <Text style={styles.resultStatValue}>{items.length}</Text>
              <Text style={styles.resultStatLabel}>总计</Text>
            </View>
          </View>
        )}

        {/* 操作按钮 */}
        <View style={styles.actionSection}>
          {batchPhase === 'config' && (
            <TouchableOpacity
              style={[styles.startBtn, (items.length === 0 || !selectedAlgId) && styles.startBtnDisabled]}
              onPress={handleStartBatch}
              disabled={items.length === 0 || !selectedAlgId}
              activeOpacity={0.8}
            >
              <Icon name="bolt" size={18} color="#fff" />
              <Text style={styles.startBtnText}>开始批量处理</Text>
            </TouchableOpacity>
          )}
          {batchPhase === 'done' && (
            <TouchableOpacity style={styles.resetBtn} onPress={handleReset} activeOpacity={0.7}>
              <Icon name="refresh" size={16} color={theme.colors.text.secondary} />
              <Text style={styles.resetBtnText}>重新开始</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.background.secondary },
  scroll: { flex: 1 },
  scrollContent: { padding: theme.spacing.md, paddingBottom: theme.spacing.xxxl },
  section: { marginBottom: theme.spacing.md },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.sm },
  sectionTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary, marginBottom: theme.spacing.sm },
  imageGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: theme.spacing.sm },
  imageItem: { width: '30%', aspectRatio: 1, borderRadius: theme.layout.borderRadius.md, overflow: 'hidden', backgroundColor: theme.colors.background.tertiary },
  thumbImage: { width: '100%', height: '100%' },
  removeBtn: { position: 'absolute', top: 4, right: 4, width: 22, height: 22, borderRadius: 11, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center' },
  statusBadge: { position: 'absolute', bottom: 4, left: 4, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4 },
  statusDone: { backgroundColor: theme.colors.status.success },
  statusFailed: { backgroundColor: theme.colors.status.error },
  statusProcessing: { backgroundColor: theme.colors.status.warning },
  statusBadgeText: { fontSize: 9, color: '#fff', fontWeight: theme.typography.weights.semibold },
  viewResultBtn: { position: 'absolute', bottom: 4, right: 4, flexDirection: 'row', alignItems: 'center', gap: 2, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4, backgroundColor: theme.colors.primary },
  retryBtn: { position: 'absolute', bottom: 4, right: 4, flexDirection: 'row', alignItems: 'center', gap: 2, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4, backgroundColor: theme.colors.status.warning },
  viewResultText: { fontSize: 9, color: '#fff', fontWeight: theme.typography.weights.semibold },
  uploadPlaceholder: { height: 120, borderRadius: theme.layout.borderRadius.lg, borderWidth: 2, borderColor: theme.colors.border.light, borderStyle: 'dashed', justifyContent: 'center', alignItems: 'center', backgroundColor: theme.colors.background.primary, gap: theme.spacing.sm },
  uploadPlaceholderText: { fontSize: theme.typography.sizes.bodySmall, color: theme.colors.text.tertiary },
  addBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  addBtnText: { fontSize: theme.typography.sizes.small, color: theme.colors.primary, fontWeight: theme.typography.weights.medium },
  algScrollContent: { gap: theme.spacing.sm },
  algChip: { paddingHorizontal: theme.spacing.md, paddingVertical: theme.spacing.sm, borderRadius: theme.layout.borderRadius.full, backgroundColor: theme.colors.background.primary, borderWidth: 1, borderColor: theme.colors.border.light },
  algChipSelected: { backgroundColor: `${theme.colors.primary}15`, borderColor: theme.colors.primary },
  algChipText: { fontSize: theme.typography.sizes.small, color: theme.colors.text.secondary, maxWidth: 120 },
  algChipTextSelected: { color: theme.colors.primary, fontWeight: theme.typography.weights.semibold },
  emptyAlgText: { fontSize: theme.typography.sizes.small, color: theme.colors.text.tertiary, fontStyle: 'italic' },
  paramsInput: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.md, borderWidth: 1, borderColor: theme.colors.border.light, padding: theme.spacing.md, fontSize: theme.typography.sizes.bodySmall, fontFamily: 'Menlo', color: theme.colors.text.primary, minHeight: 100, textAlignVertical: 'top' },
  paramsInputError: { borderColor: theme.colors.status.error },
  paramsErrorText: { fontSize: theme.typography.sizes.small, color: theme.colors.status.error, marginTop: 4 },
  progressSection: { backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.lg, marginBottom: theme.spacing.md, ...theme.layout.shadows.sm },
  progressHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: theme.spacing.sm },
  progressTitle: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.semibold, color: theme.colors.text.primary },
  progressText: { fontSize: theme.typography.sizes.medium, fontWeight: theme.typography.weights.bold, color: theme.colors.primary },
  progressTrack: { height: 10, backgroundColor: theme.colors.background.tertiary, borderRadius: 5, overflow: 'hidden', marginBottom: theme.spacing.md },
  progressFill: { height: '100%', backgroundColor: theme.colors.primary, borderRadius: 5 },
  cancelBtn: { alignSelf: 'center', paddingHorizontal: theme.spacing.lg, paddingVertical: theme.spacing.sm, borderRadius: theme.layout.borderRadius.md, borderWidth: 1, borderColor: theme.colors.status.error },
  cancelBtnText: { fontSize: theme.typography.sizes.medium, color: theme.colors.status.error, fontWeight: theme.typography.weights.medium },
  resultSummary: { flexDirection: 'row', backgroundColor: theme.colors.background.primary, borderRadius: theme.layout.borderRadius.lg, padding: theme.spacing.lg, marginBottom: theme.spacing.md, ...theme.layout.shadows.sm },
  resultStat: { flex: 1, alignItems: 'center' },
  resultStatValue: { fontSize: 28, fontWeight: theme.typography.weights.bold },
  resultStatLabel: { fontSize: theme.typography.sizes.small, color: theme.colors.text.secondary, marginTop: 2 },
  resultStatDivider: { width: 1, backgroundColor: theme.colors.border.light, marginVertical: 8 },
  actionSection: { marginTop: theme.spacing.md },
  startBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: theme.spacing.sm, paddingVertical: theme.spacing.lg, borderRadius: theme.layout.borderRadius.md, backgroundColor: theme.colors.primary, ...theme.layout.shadows.sm },
  startBtnDisabled: { opacity: 0.5 },
  startBtnText: { fontSize: theme.typography.sizes.large, fontWeight: theme.typography.weights.bold, color: '#fff' },
  resetBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: theme.spacing.sm, paddingVertical: theme.spacing.md, borderRadius: theme.layout.borderRadius.md, borderWidth: 1, borderColor: theme.colors.border.light },
  resetBtnText: { fontSize: theme.typography.sizes.medium, color: theme.colors.text.secondary, fontWeight: theme.typography.weights.medium },
});

export default BatchScreen;
