/**
 * 去雾处理流程 hook
 *
 * 仅封装 dehaze 页与 processing 页真正同源的 predict/cancel/retry 三个操作
 * 与 cancelSignalRef 的协同逻辑。不接管 UI 渲染、historyStorage 写入、
 * "确认开始" 弹窗等页面差异逻辑。
 */
import { useCallback, useRef, useState } from 'react';
import { Alert } from 'react-native';
import type { Algorithm } from 'dehaze-sdk-js';
import type { SelectedImage } from '@/types/image';
import type {
  CommonAlgorithmParams,
  ProcessingResult,
  TaskProgress,
} from '@/types/processing';
import { predictSingle } from '@/pages/processing/services/processingApi';

export type ProcessingPhase = 'config' | 'processing' | 'done' | 'failed';

interface ProcessingState {
  phase: ProcessingPhase;
  progress: TaskProgress | null;
  result: ProcessingResult | null;
  predict: (
    image: SelectedImage,
    algorithm: Algorithm,
    params?: CommonAlgorithmParams,
  ) => Promise<void>;
  cancel: () => Promise<void>;
  retry: () => Promise<void>;
  reset: () => void;
}

export function useProcessing(): ProcessingState {
  const [phase, setPhase] = useState<ProcessingPhase>('config');
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);

  // 取消信号（用 ref 避免闭包陈旧）
  const cancelSignalRef = useRef<{ canceled: boolean }>({ canceled: false });
  // 记录最近一次 predict 入参，供 retry 复用
  const lastPredictRef = useRef<{
    image: SelectedImage;
    algorithm: Algorithm;
    params?: CommonAlgorithmParams;
  } | null>(null);

  const predict = useCallback(
    async (
      image: SelectedImage,
      algorithm: Algorithm,
      params?: CommonAlgorithmParams,
    ) => {
      if (!image?.url || !algorithm?.id) return;

      lastPredictRef.current = { image, algorithm, params };
      cancelSignalRef.current = { canceled: false };
      setPhase('processing');
      setProgress({ status: 0, elapsed: 0 });
      setResult(null);

      try {
        const res = await predictSingle({
          algorithmId: algorithm.id,
          imageUrl: image.url,
          params,
          onProgress: p => setProgress(p),
          cancelSignal: cancelSignalRef.current,
        });
        setResult(res);
        setPhase('done');
      } catch (err) {
        const isCanceled = err instanceof Error && err.message.includes('取消');
        setProgress(prev => ({
          status: isCanceled ? 4 : 3,
          elapsed: prev?.elapsed ?? 0,
          error: err instanceof Error ? err.message : '处理失败',
        }));
        setPhase(isCanceled ? 'config' : 'failed');
        if (!isCanceled) {
          Alert.alert('处理失败', err instanceof Error ? err.message : '请稍后重试');
        }
      }
    },
    [],
  );

  const cancel = useCallback(async () => {
    await new Promise<void>(resolve => {
      Alert.alert('确认取消', '确定要取消当前处理任务吗？', [
        { text: '继续处理', style: 'cancel', onPress: () => resolve() },
        {
          text: '取消处理',
          style: 'destructive',
          onPress: () => {
            cancelSignalRef.current.canceled = true;
            resolve();
          },
        },
      ]);
    });
  }, []);

  const retry = useCallback(async () => {
    const last = lastPredictRef.current;
    if (!last) return;
    await predict(last.image, last.algorithm, last.params);
  }, [predict]);

  const reset = useCallback(() => {
    cancelSignalRef.current = { canceled: false };
    setPhase('config');
    setProgress(null);
    setResult(null);
  }, []);

  return { phase, progress, result, predict, cancel, retry, reset };
}