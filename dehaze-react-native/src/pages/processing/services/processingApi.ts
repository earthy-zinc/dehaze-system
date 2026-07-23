/**
 * 去雾处理服务
 *
 * 封装 dehaze-sdk-js 的 ModelAPI：
 * - predict: 执行去雾预测（同步返回结果）
 * - getPredTaskStatus: 查询任务状态（仅当 predict 未直接返回结果时轮询）
 *
 * 提供：
 * - 单张处理 + 任务状态轮询
 * - 批量处理（串行）
 *
 * 说明：API 同步返回处理结果，前端不模拟阶段化进度百分比，
 *       仅展示真实处理状态与已用时间。
 */
import { ModelAPI } from 'dehaze-sdk-js';
import type { PredictionResultVO } from 'dehaze-sdk-js';
import type {
  CommonAlgorithmParams,
  ProcessingResult,
  TaskProgress,
} from '@/types/processing';

/** 默认参数（推荐配置） */
export const DEFAULT_PARAMS: CommonAlgorithmParams = {
  strength: 50,
  saturation: 100,
  contrast: 100,
  sharpen: 30,
};

/** 参数 schema（用于动态渲染调节面板） */
export const PARAM_SCHEMAS = [
  {
    key: 'strength' as const,
    label: '去雾强度',
    type: 'slider' as const,
    min: 0,
    max: 100,
    step: 1,
    default: 50,
    description: '控制去雾程度，值越大去雾效果越强',
  },
  {
    key: 'saturation' as const,
    label: '色彩饱和度',
    type: 'slider' as const,
    min: 0,
    max: 200,
    step: 1,
    default: 100,
    description: '调整色彩鲜艳度，100 为原始值',
  },
  {
    key: 'contrast' as const,
    label: '对比度',
    type: 'slider' as const,
    min: 0,
    max: 200,
    step: 1,
    default: 100,
    description: '调整明暗对比，100 为原始值',
  },
  {
    key: 'sharpen' as const,
    label: '锐化程度',
    type: 'slider' as const,
    min: 0,
    max: 100,
    step: 1,
    default: 30,
    description: '增强细节清晰度',
  },
];

/** 预设方案 */
export const PARAM_PRESETS = [
  {
    key: 'recommended',
    name: '推荐配置',
    description: '适用于大多数场景的均衡配置',
    params: { strength: 50, saturation: 100, contrast: 100, sharpen: 30 },
  },
  {
    key: 'landscape',
    name: '风景优化',
    description: '增强色彩与对比，适合户外风景',
    params: { strength: 70, saturation: 130, contrast: 120, sharpen: 40 },
  },
  {
    key: 'night',
    name: '夜景增强',
    description: '降低强度，保留夜景氛围',
    params: { strength: 35, saturation: 90, contrast: 110, sharpen: 50 },
  },
];

/** 轮询间隔（ms） */
const POLL_INTERVAL = 1500;
/** 最大轮询时长（ms） */
const MAX_POLL_DURATION = 5 * 60 * 1000;

/** 将参数对象序列化为后端可识别的 JSON 字符串 */
function serializeParams(params?: CommonAlgorithmParams): string | undefined {
  if (!params) return undefined;
  // 仅保留已设置字段
  const filtered: Record<string, number> = {};
  (Object.keys(params) as (keyof CommonAlgorithmParams)[]).forEach(k => {
    const v = params[k];
    if (v !== undefined && v !== null) {
      filtered[k] = v;
    }
  });
  return Object.keys(filtered).length ? JSON.stringify(filtered) : undefined;
}

/** 将 SDK 预测结果映射为前端处理结果 */
function toProcessingResult(vo: PredictionResultVO): ProcessingResult {
  return {
    logId: vo.logId,
    resultUrl: vo.resultUrl,
    resultThumbnailUrl: vo.resultThumbnailUrl,
    time: vo.time,
    fromCache: vo.fromCache,
  };
}

export interface PredictOptions {
  algorithmId: number;
  imageUrl: string;
  params?: CommonAlgorithmParams;
  /** 进度回调（用于真实状态更新 + 轮询状态） */
  onProgress?: (progress: TaskProgress) => void;
  /** 取消信号 */
  cancelSignal?: { canceled: boolean };
}

/**
 * 单张去雾处理
 *
 * 1. 调用 ModelAPI.predict 提交预测任务
 * 2. 若返回结果含 resultUrl，直接完成
 * 3. 若仅返回 logId 无 resultUrl，启动轮询 getPredTaskStatus
 *
 * 不模拟进度百分比，仅推进真实状态与已用时间。
 */
export async function predictSingle(opts: PredictOptions): Promise<ProcessingResult> {
  const { algorithmId, imageUrl, params, onProgress, cancelSignal } = opts;
  const startTime = Date.now();

  const emit = (status: TaskProgress['status'], error?: string) => {
    onProgress?.({
      status,
      elapsed: Date.now() - startTime,
      error,
    });
  };

  // 进入处理中状态
  emit('processing');

  // 提交预测任务
  let result: PredictionResultVO;
  try {
    result = await ModelAPI.predict({
      algorithmId,
      imageUrl,
      params: serializeParams(params),
    });
  } catch (err) {
    emit('failed', err instanceof Error ? err.message : '预测请求失败');
    throw err;
  }

  if (cancelSignal?.canceled) throw new Error('用户已取消处理');

  // 若返回结果无 resultUrl，启动轮询
  if (!result.resultUrl && result.logId) {
    result = await pollTaskStatus(result.logId, startTime, onProgress, cancelSignal);
  }

  emit('success');
  return toProcessingResult(result);
}

/**
 * 轮询任务状态
 *
 * @param taskId 后端返回的 logId
 * @param startTime 任务开始时间（用于超时判定）
 */
async function pollTaskStatus(
  taskId: number,
  startTime: number,
  onProgress?: (p: TaskProgress) => void,
  cancelSignal?: { canceled: boolean },
): Promise<PredictionResultVO> {
  while (true) {
    if (cancelSignal?.canceled) throw new Error('用户已取消处理');

    const elapsed = Date.now() - startTime;
    if (elapsed > MAX_POLL_DURATION) {
      throw new Error('处理超时，请稍后重试');
    }

    // 仅推进真实已用时间，不模拟百分比
    onProgress?.({
      status: 'processing',
      elapsed,
    });

    try {
      const status = await ModelAPI.getPredTaskStatus(taskId);
      if (status.resultUrl) {
        // 拿到最终结果
        return status;
      }
    } catch {
      // 轮询出错时继续重试，不立即失败
    }

    await delay(POLL_INTERVAL);
  }
}

/** 批量处理（串行执行） */
export interface BatchPredictOptions {
  algorithmId: number;
  images: { url: string; name?: string }[];
  params?: CommonAlgorithmParams;
  /** 单张完成回调 */
  onItemComplete?: (index: number, result: ProcessingResult) => void;
  /** 单张失败回调 */
  onItemError?: (index: number, error: Error) => void;
  /** 单张进度回调 */
  onItemProgress?: (index: number, progress: TaskProgress) => void;
  /** 取消信号 */
  cancelSignal?: { canceled: boolean };
}

export async function predictBatch(opts: BatchPredictOptions): Promise<ProcessingResult[]> {
  const { algorithmId, images, params, onItemComplete, onItemError, onItemProgress, cancelSignal } = opts;
  const results: ProcessingResult[] = [];

  for (let i = 0; i < images.length; i++) {
    if (cancelSignal?.canceled) break;

    try {
      const result = await predictSingle({
        algorithmId,
        imageUrl: images[i].url,
        params,
        onProgress: p => onItemProgress?.(i, p),
        cancelSignal,
      });
      results.push(result);
      onItemComplete?.(i, result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('处理失败');
      onItemError?.(i, error);
      // 失败不阻塞后续任务
    }
  }

  return results;
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
